#include "pinocchio_cppadcg.hh"

#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/parsers/srdf.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/geometry.hpp>
#include <pinocchio/multibody/geometry.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/collision/collision.hpp>

#include <coal/shape/geometric_shapes.h>

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Min_sphere_of_spheres_d.h>
#include <CGAL/Min_sphere_of_spheres_d_traits_3.h>

#include <fmt/core.h>
#include <nlohmann/json.hpp>
#include <inja/inja.hpp>
#include <cxxopts.hpp>

#include <filesystem>
#include <stdexcept>
#include <vector>
#include <optional>

#include "lang_cpp.hh"
#include "lang_rust.hh"

#include <chrono>

#include <cppad/cg/model/model_library_c_source_gen.hpp>
#include <cppad/cg/model/compiler/gcc_compiler.hpp>
#include <cppad/cg/model/dynamic_lib/linux/linux_dynamic_model_library_processor.hpp>

using namespace pinocchio;
using namespace CppAD;
using namespace CppAD::cg;

// Typedef for AD types
using CGD = CG<double>;
using ADCG = AD<CGD>;

using ADModel = ModelTpl<ADCG>;
using ADData = DataTpl<ADCG>;
using ADVectorXs = Eigen::Matrix<ADCG, Eigen::Dynamic, 1>;

struct SphereInfo
{
    std::size_t geom_index;
    float radius;
    std::size_t parent_joint;
    std::size_t parent_frame;
    SE3 relative;
};

auto min_sphere_of_spheres(const std::vector<SphereInfo> &info) -> std::array<float, 4>
{
    using K = CGAL::Exact_predicates_inexact_constructions_kernel;
    using Traits = CGAL::Min_sphere_of_spheres_d_traits_3<K, double>;
    using Sphere = Traits::Sphere;
    using Point = K::Point_3;
    using MinSphere = CGAL::Min_sphere_of_spheres_d<Traits>;

    std::vector<Sphere> cgal_spheres;
    cgal_spheres.reserve(info.size());

    for (const auto &sphere : info)
    {
        auto pos = sphere.relative.translation();
        cgal_spheres.emplace_back(Point(pos[0], pos[1], pos[2]), sphere.radius);
    }

    MinSphere ms(cgal_spheres.begin(), cgal_spheres.end());
    std::array<float, 4> sphere;
    std::copy(ms.center_cartesian_begin(), ms.center_cartesian_end(), sphere.begin());
    sphere[3] = ms.radius();
    return sphere;
}

struct RobotInfo
{
    RobotInfo(
        const std::filesystem::path &urdf_file,
        const std::optional<std::filesystem::path> &srdf_file,
        const std::optional<std::string> &end_effector)
    {
        if (not std::filesystem::exists(urdf_file))
        {
            throw std::runtime_error(fmt::format("URDF file {} does not exist!", urdf_file.string()));
        }

        pinocchio::urdf::buildModel(urdf_file, model);
        pinocchio::urdf::buildGeom(model, urdf_file, COLLISION, collision_model);

        if (srdf_file and not std::filesystem::exists(*srdf_file))
        {
            throw std::runtime_error(fmt::format("SRDF file () does not exist!", srdf_file->string()));
        }
        else if (not srdf_file)
        {
            fmt::print("No SRDF file provided, guessing collisions!\n");
            guess_self_collisions();
        }
        else
        {
            collision_model.addAllCollisionPairs();
            pinocchio::srdf::removeCollisionPairs(model, collision_model, *srdf_file);
            extract_collision_data();
        }

        extract_spheres();

        if (not end_effector)
        {
            end_effector_name = model.frames[model.nframes - 1].name;
            fmt::print("No EE provided, using distal link `{}`.\n", end_effector_name);
        }
        else if (not model.existFrame(*end_effector))
        {
            throw std::runtime_error(fmt::format("Invalid EE name {}", *end_effector));
        }
        else
        {
            end_effector_name = *end_effector;
        }

        end_effector_index = model.getFrameId(end_effector_name);
    }

    auto json() -> nlohmann::json
    {
        const Eigen::VectorXd lower_bound = model.lowerPositionLimit;
        const Eigen::VectorXd upper_bound = model.upperPositionLimit;
        const Eigen::VectorXd bound_range = upper_bound - lower_bound;
        const Eigen::VectorXd bound_descale = bound_range.cwiseInverse();

        nlohmann::json json;
        json["n_q"] = model.nq;
        json["n_spheres"] = spheres.size();
        json["bound_lower"] = std::vector<float>(lower_bound.data(), lower_bound.data() + model.nq);
        json["bound_range"] = std::vector<float>(bound_range.data(), bound_range.data() + model.nq);
        json["bound_descale"] = std::vector<float>(bound_descale.data(), bound_descale.data() + model.nq);
        json["measure"] = bound_range.prod();
        json["end_effector"] = end_effector_name;
        json["end_effector_index"] = end_effector_index;
        json["min_radius"] = min_radius;
        json["max_radius"] = max_radius;
        json["joint_names"] = dof_to_joint_names();
        json["allowed_link_pairs"] = allowed_link_pairs;
        json["per_link_spheres"] = per_link_spheres;
        json["links_with_geometry"] = links_with_geometry;
        json["bounding_sphere_index"] = bounding_sphere_index;
        json["end_effector_collisions"] = get_frames_colliding_end_effector();

        std::vector<std::string> link_names;
        for (auto i = 0U; i < model.frames.size(); ++i)
        {
            link_names.emplace_back(model.frames[i].name);
        }
        json["link_names"] = link_names;

        return json;
    }

    auto dof_to_joint_names() -> std::vector<std::string>
    {
        std::vector<std::size_t> dof_to_joint_id(model.nq);
        for (auto joint_id = 1U; joint_id < model.joints.size(); ++joint_id)
        {
            const auto &joint = model.joints[joint_id];
            auto start_idx = joint.idx_q();
            auto nq = joint.nq();

            for (auto i = 0U; i < nq; ++i)
            {
                dof_to_joint_id[start_idx + i] = joint_id;
            }
        }

        std::vector<std::string> dof_to_joint_name(model.nq);
        for (auto i = 0U; i < model.nq; ++i)
        {
            dof_to_joint_name[i] = model.names[dof_to_joint_id[i]];
        }

        return dof_to_joint_name;
    }

    auto get_frames_colliding_end_effector() -> std::vector<std::size_t>
    {
        std::size_t end_effector_joint = model.frames[end_effector_index].parentJoint;

        std::vector<std::size_t> frames;
        for (auto i = 0U; i < model.frames.size(); ++i)
        {
            if (model.frames[i].parentJoint == end_effector_joint)
            {
                if (bounding_spheres.find(i) != bounding_spheres.end())
                {
                    frames.emplace_back(i);
                }
            }
        }

        std::set<std::size_t> end_effector_allowed_collisions;
        for (const auto &[first, second] : allowed_link_pairs)
        {
            if (std::find(frames.begin(), frames.end(), first) != frames.end())
            {
                end_effector_allowed_collisions.emplace(second);
            }

            if (std::find(frames.begin(), frames.end(), second) != frames.end())
            {
                end_effector_allowed_collisions.emplace(first);
            }
        }

        return std::vector<std::size_t>(
            end_effector_allowed_collisions.begin(), end_effector_allowed_collisions.end());
    }

    auto extract_spheres() -> void
    {
        for (auto i = 0U; i < collision_model.ngeoms; ++i)
        {
            const auto &geom_obj = collision_model.geometryObjects[i];
            const auto &sphere_ptr = std::dynamic_pointer_cast<coal::Sphere>(geom_obj.geometry);

            if (sphere_ptr)
            {
                SphereInfo info;
                info.geom_index = i;
                info.radius = sphere_ptr->radius;
                info.parent_joint = geom_obj.parentJoint;
                info.parent_frame = geom_obj.parentFrame;
                info.relative = geom_obj.placement;

                spheres.emplace_back(info);

                min_radius = std::min(min_radius, info.radius);
                max_radius = std::max(max_radius, info.radius);
            }
            else
            {
                throw std::runtime_error(
                    fmt::format("Invalid non-sphere geometry in URDF {}", geom_obj.name));
            }
        }

        std::size_t bs = 0;
        for (auto i = 0U; i < model.frames.size(); ++i)
        {
            std::vector<SphereInfo> link_info;
            std::vector<std::size_t> sphere_indices;
            for (const auto &info : spheres)
            {
                if (info.parent_frame == i)
                {
                    link_info.emplace_back(info);
                    sphere_indices.emplace_back(info.geom_index);
                }
            }

            per_link_spheres.emplace_back(sphere_indices);

            if (not link_info.empty())
            {
                auto sphere = min_sphere_of_spheres(link_info);

                SphereInfo info;
                info.geom_index = bs;
                info.radius = sphere[3];
                info.parent_joint = link_info[0].parent_joint;
                info.relative = SE3::Identity();
                info.relative.translation()[0] = sphere[0];
                info.relative.translation()[1] = sphere[1];
                info.relative.translation()[2] = sphere[2];

                bounding_spheres.emplace(i, info);
                bounding_sphere_index.emplace_back(bs);
                links_with_geometry.emplace_back(i);
                bs++;
            }
            else
            {
                bounding_sphere_index.emplace_back(0);
            }
        }
    }

    auto collision_pair_to_frame_pair(const CollisionPair &cp) -> std::pair<std::size_t, std::size_t>
    {
        const auto &geom1 = collision_model.geometryObjects[cp.first];
        const auto &geom2 = collision_model.geometryObjects[cp.second];

        std::size_t link1_idx = geom1.parentFrame;
        std::size_t link2_idx = geom2.parentFrame;

        return std::make_pair(std::min(link1_idx, link2_idx), std::max(link1_idx, link2_idx));
    }

    auto extract_collision_data() -> void
    {
        for (const auto &cp : collision_model.collisionPairs)
        {
            allowed_link_pairs.insert(collision_pair_to_frame_pair(cp));
        }
    }

    auto get_adjacent_frames() -> std::set<std::pair<std::size_t, std::size_t>>
    {
        const auto nf = model.frames.size();
        const auto nj = model.joints.size();

        std::set<std::pair<std::size_t, std::size_t>> adjacents;

        for (auto i = 0U; i < nf; ++i)
        {
            for (auto j = i + 1; j < nf; ++j)
            {
                const auto &frame_i = model.frames[i];
                const auto &frame_j = model.frames[j];

                if (frame_i.parentJoint < nj and frame_j.parentJoint < nj)
                {
                    const auto &joint_i = model.joints[frame_i.parentJoint];
                    const auto &joint_j = model.joints[frame_j.parentJoint];

                    // Check if joints are parent-child related
                    if (model.parents[frame_i.parentJoint] == frame_j.parentJoint or
                        model.parents[frame_j.parentJoint] == frame_i.parentJoint)
                    {
                        adjacents.insert({i, j});
                    }
                }
            }
        }

        return adjacents;
    }

    auto guess_self_collisions(std::size_t n = 1000000U) -> void
    {
        collision_model.addAllCollisionPairs();

        Data data(model);
        GeometryData collision_data(collision_model);

        std::set<std::pair<std::size_t, std::size_t>> always_pairs;

        for (auto j = 0U; j < collision_model.collisionPairs.size(); ++j)
        {
            always_pairs.emplace(collision_pair_to_frame_pair(collision_model.collisionPairs[j]));
        }

        allowed_link_pairs.clear();

        for (auto i = 0U; i < n; ++i)
        {
            auto q = randomConfiguration(model);
            computeCollisions(model, data, collision_model, collision_data, q);

            for (auto j = 0U; j < collision_model.collisionPairs.size(); ++j)
            {
                const auto &cr = collision_data.collisionResults[j];
                auto pair = collision_pair_to_frame_pair(collision_model.collisionPairs[j]);

                if (cr.isCollision())
                {
                    allowed_link_pairs.insert(pair);
                }
                else
                {
                    auto it = always_pairs.find(pair);
                    if (it != always_pairs.end())
                    {
                        always_pairs.erase(it);
                    }
                }
            }
        }

        // Remove all adjacent frames
        auto adjacents = get_adjacent_frames();
        for (const auto &pair : adjacents)
        {
            allowed_link_pairs.erase(pair);
        }

        // Remove all pairs that never collided
        for (const auto &pair : always_pairs)
        {
            allowed_link_pairs.erase(pair);
        }

        // Add remaining potential collisions
        collision_model.removeAllCollisionPairs();
        for (const auto &pair : allowed_link_pairs)
        {
            collision_model.addCollisionPair(CollisionPair(pair.first, pair.second));
        }
    }

    Model model;
    GeometryModel collision_model;
    std::string end_effector_name;
    std::size_t end_effector_index;

    float min_radius{std::numeric_limits<float>::max()};
    float max_radius{std::numeric_limits<float>::min()};
    std::vector<SphereInfo> spheres;
    std::map<std::size_t, SphereInfo> bounding_spheres;
    std::vector<std::size_t> links_with_geometry;
    std::vector<std::vector<std::size_t>> per_link_spheres;
    std::set<std::pair<std::size_t, std::size_t>> allowed_link_pairs;
    std::vector<std::size_t> bounding_sphere_index;
};

auto trace_sphere(const SphereInfo &sphere, const ADData &ad_data, ADVectorXs &data, std::size_t index)
{
    const auto &joint_placement = ad_data.oMi[sphere.parent_joint];

    Eigen::Matrix<ADCG, 3, 1> local_translation;
    local_translation[0] = sphere.relative.translation()[0];
    local_translation[1] = sphere.relative.translation()[1];
    local_translation[2] = sphere.relative.translation()[2];

    Eigen::Matrix<ADCG, 3, 1> world_position =
        joint_placement.rotation() * local_translation + joint_placement.translation();

    data[index + 0] = world_position[0];
    data[index + 1] = world_position[1];
    data[index + 2] = world_position[2];
    data[index + 3] = ADCG(sphere.radius);
}

auto trace_frame(std::size_t ee_index, const ADData &ad_data, ADVectorXs &data, std::size_t index)
{
    const auto &oMf = ad_data.oMf[ee_index];

    data[index + 0] = oMf.translation()[0];
    data[index + 1] = oMf.translation()[1];
    data[index + 2] = oMf.translation()[2];

    const auto &R = oMf.rotation();

    // Eigen stores as column major
    data[index + 3] = R(0, 0);
    data[index + 4] = R(1, 0);
    data[index + 5] = R(2, 0);
    data[index + 6] = R(0, 1);
    data[index + 7] = R(1, 1);
    data[index + 8] = R(2, 1);
    data[index + 9] = R(0, 2);
    data[index + 10] = R(1, 2);
    data[index + 11] = R(2, 2);
}

struct Traced
{
    std::string code;
    std::size_t temp_variables;
    std::size_t outputs;
};

auto trace_sphere_cc_fk(
    const RobotInfo &info,
    const std::string &language,
    bool spheres = true,
    bool bounding_spheres = true,
    bool fk = true) -> Traced
{
    auto nq = info.model.nq;
    ADModel ad_model = info.model.cast<ADCG>();
    ADData ad_data(ad_model);

    ADVectorXs ad_q(nq);
    for (auto i = 0U; i < nq; ++i)
    {
        ad_q[i] = ADCG(0.0);
    }

    Independent(ad_q);

    forwardKinematics(ad_model, ad_data, ad_q);
    updateFramePlacements(ad_model, ad_data);

    std::size_t n_spheres_data = (spheres) ? info.spheres.size() * 4 : 0;
    std::size_t n_bounding_spheres_data = (bounding_spheres) ? info.bounding_spheres.size() * 4 : 0;
    std::size_t n_fk_data = (fk) ? 12 : 0;
    std::size_t n_out = n_spheres_data + n_bounding_spheres_data + n_fk_data;

    ADVectorXs data(n_out);

    if (spheres)
    {
        for (auto i = 0U; i < info.spheres.size(); ++i)
        {
            const auto &sphere = info.spheres[i];
            trace_sphere(sphere, ad_data, data, sphere.geom_index * 4);
        }
    }

    if (bounding_spheres)
    {
        for (auto i = 0U; i < info.model.frames.size(); ++i)
        {
            auto sphere_it = info.bounding_spheres.find(i);
            if (sphere_it != info.bounding_spheres.end())
            {
                const auto &sphere = sphere_it->second;
                trace_sphere(sphere, ad_data, data, sphere.geom_index * 4 + n_spheres_data);
            }
        }
    }

    if (fk)
    {
        trace_frame(info.end_effector_index, ad_data, data, n_spheres_data + n_bounding_spheres_data);
    }

    // Create the AD function
    ADFun<CGD> collision_sphere_func(ad_q, data);

    CodeHandler<double> handler;
    CppAD::vector<CGD> ind_vars(nq);
    handler.makeVariables(ind_vars);

    CppAD::vector<CGD> result = collision_sphere_func.Forward(0, ind_vars);

    LangCDefaultVariableNameGenerator<double> nameGen;
    std::ostringstream function_code;

    if (language == "c++")
    {
        LanguageCCustom<double> langC("double");
        handler.generateCode(function_code, langC, result, nameGen);
    }
    else if (language == "rust")
    {
        LanguageRust<double> langRust("double");
        handler.generateCode(function_code, langRust, result, nameGen);
    }
    else
    {
        throw std::runtime_error(fmt::format("unsupported language {}", language));
    }

    return Traced{function_code.str(), handler.getTemporaryVariableCount(), n_out};
}

auto trace_collision_gradient(
    const RobotInfo &info,
    const std::string &language,
    bool spheres = true,
    bool bounding_spheres = true) -> Traced
{
    auto nq = info.model.nq;
    auto nv = info.model.nv;

    ADModel ad_model = info.model.cast<ADCG>();
    ADData ad_data(ad_model);

    std::size_t n_spheres_count = 0;
    if (spheres)
    {
        n_spheres_count += info.spheres.size();
    }
    if (bounding_spheres)
    {
        n_spheres_count += info.bounding_spheres.size();
    }

    // Input vector: [q (nq) | gradients (3 * n_spheres)]
    std::size_t input_dim = nq + 3 * n_spheres_count;
    ADVectorXs ad_x(input_dim);
    for (auto i = 0U; i < input_dim; ++i)
    {
        ad_x[i] = ADCG(0.0);
    }

    Independent(ad_x);

    // Extract q and gradients
    ADVectorXs ad_q = ad_x.head(nq);

    // Forward Kinematics & Jacobians
    pinocchio::computeJointJacobians(ad_model, ad_data, ad_q);
    pinocchio::framesForwardKinematics(ad_model, ad_data, ad_q);

    // Prepare output gradient (generalized forces in tangent space)
    ADVectorXs total_grad = ADVectorXs::Zero(nv);

    // Using Data::Matrix6x for Jacobian storage
    pinocchio::DataTpl<ADCG>::Matrix6x J_joint(6, nv);

    std::size_t grad_idx = nq;  // Start reading gradients after q

    auto process_sphere = [&](const SphereInfo &sphere) {
        // 1. Read input gradient for this sphere
        Eigen::Matrix<ADCG, 3, 1> sdf_grad;
        sdf_grad[0] = ad_x[grad_idx++];
        sdf_grad[1] = ad_x[grad_idx++];
        sdf_grad[2] = ad_x[grad_idx++];

        // 2. Compute Joint Jacobian (Local World Aligned)
        // This gives us the spatial velocity of the Joint Origin, expressed in World Frame.
        J_joint.setZero();
        pinocchio::getJointJacobian(
            ad_model, ad_data, sphere.parent_joint, pinocchio::LOCAL_WORLD_ALIGNED, J_joint);

        // 3. Compute sphere offset in World Frame
        // sphere.relative is in Joint Frame.
        const auto &oMi = ad_data.oMi[sphere.parent_joint];
        Eigen::Matrix<ADCG, 3, 1> offset_local;
        offset_local[0] = sphere.relative.translation()[0];
        offset_local[1] = sphere.relative.translation()[1];
        offset_local[2] = sphere.relative.translation()[2];

        Eigen::Matrix<ADCG, 3, 1> offset_world = oMi.rotation() * offset_local;

        // 4. Transform Force to Joint Origin
        // Force at sphere P: F = sdf_grad
        // Equivalent Wrench at Joint Origin O:
        //   Force F_o = F
        //   Torque Tau_o = (P - O) x F = offset_world x sdf_grad
        Eigen::Matrix<ADCG, 3, 1> torque_at_joint = offset_world.cross(sdf_grad);

        // 5. Project to Generalized Coordinates
        // tau_gen = J_joint^T * [F_o; Tau_o]
        Eigen::Matrix<ADCG, 6, 1> spatial_force;
        spatial_force.head(3) = sdf_grad;
        spatial_force.tail(3) = torque_at_joint;

        total_grad += J_joint.transpose() * spatial_force;
    };

    if (spheres)
    {
        for (const auto &sphere : info.spheres)
        {
            process_sphere(sphere);
        }
    }

    if (bounding_spheres)
    {
        // Iterate in frame order to match trace_sphere_cc_fk
        for (auto i = 0U; i < info.model.frames.size(); ++i)
        {
            auto it = info.bounding_spheres.find(i);
            if (it != info.bounding_spheres.end())
            {
                process_sphere(it->second);
            }
        }
    }

    // Generate Code
    ADFun<CGD> gradient_func(ad_x, total_grad);

    CodeHandler<double> handler;
    CppAD::vector<CGD> ind_vars(input_dim);
    handler.makeVariables(ind_vars);

    CppAD::vector<CGD> result = gradient_func.Forward(0, ind_vars);

    LangCDefaultVariableNameGenerator<double> nameGen;
    std::ostringstream function_code;

    if (language == "c++")
    {
        LanguageCCustom<double> langC("double");
        handler.generateCode(function_code, langC, result, nameGen);
    }
    else if (language == "rust")
    {
        LanguageRust<double> langRust("double");
        handler.generateCode(function_code, langRust, result, nameGen);
    }
    else
    {
        throw std::runtime_error(fmt::format("unsupported language {}", language));
    }

    return Traced{function_code.str(), handler.getTemporaryVariableCount(), (std::size_t)nv};
}

auto verify_collision_gradient_correctness(const RobotInfo &info) -> void
{
    // 1. Setup CppAD Tape (same logic as trace_collision_gradient)
    auto nq = info.model.nq;
    auto nv = info.model.nv;

    // We use CppAD::cg::CG<double> for code generation/tracing
    using CGD = CppAD::cg::CG<double>;
    using ADCG = CppAD::AD<CGD>;
    using ADModel = pinocchio::ModelTpl<ADCG>;
    using ADData = pinocchio::DataTpl<ADCG>;
    using ADVectorXs = Eigen::Matrix<ADCG, Eigen::Dynamic, 1>;

    ADModel ad_model = info.model.cast<ADCG>();
    ADData ad_data(ad_model);

    std::size_t n_spheres_count = info.spheres.size() + info.bounding_spheres.size();
    std::size_t input_dim = nq + 3 * n_spheres_count;
    
    ADVectorXs ad_x(input_dim);
    for (auto i = 0U; i < input_dim; ++i) ad_x[i] = ADCG(0.0);

    CppAD::Independent(ad_x);

    // --- Trace Logic Start ---
    ADVectorXs ad_q = ad_x.head(nq);
    pinocchio::computeJointJacobians(ad_model, ad_data, ad_q);
    pinocchio::framesForwardKinematics(ad_model, ad_data, ad_q);
    
    ADVectorXs total_grad = ADVectorXs::Zero(nv);
    pinocchio::DataTpl<ADCG>::Matrix6x J_joint(6, nv);
    std::size_t grad_idx = nq;

    auto process_sphere = [&](const SphereInfo &sphere) {
        Eigen::Matrix<ADCG, 3, 1> sdf_grad;
        sdf_grad[0] = ad_x[grad_idx++];
        sdf_grad[1] = ad_x[grad_idx++];
        sdf_grad[2] = ad_x[grad_idx++];

        J_joint.setZero();
        pinocchio::getJointJacobian(ad_model, ad_data, sphere.parent_joint, pinocchio::LOCAL_WORLD_ALIGNED, J_joint);

        const auto &oMi = ad_data.oMi[sphere.parent_joint];
        Eigen::Matrix<ADCG, 3, 1> offset_local;
        offset_local[0] = sphere.relative.translation()[0];
        offset_local[1] = sphere.relative.translation()[1];
        offset_local[2] = sphere.relative.translation()[2];
        Eigen::Matrix<ADCG, 3, 1> offset_world = oMi.rotation() * offset_local;
        Eigen::Matrix<ADCG, 3, 1> torque_at_joint = offset_world.cross(sdf_grad);

        Eigen::Matrix<ADCG, 6, 1> spatial_force;
        spatial_force.head(3) = sdf_grad;
        spatial_force.tail(3) = torque_at_joint;

        total_grad += J_joint.transpose() * spatial_force;
    };

    for (const auto &sphere : info.spheres) process_sphere(sphere);
    for (auto i = 0U; i < info.model.frames.size(); ++i) {
        if (info.bounding_spheres.count(i)) process_sphere(info.bounding_spheres.at(i));
    }
    // --- Trace Logic End ---

    CppAD::ADFun<CGD> fun(ad_x, total_grad);

    // 2. Compile Library using CppADCodeGen
    CppAD::cg::ModelCSourceGen<double> cgen(fun, "collision_gradient");
    cgen.setCreateForwardZero(true);
    cgen.setCreateJacobian(false);
    cgen.setCreateSparseHessian(false);
    
    CppAD::cg::ModelLibraryCSourceGen<double> libcgen(cgen);
    CppAD::cg::DynamicModelLibraryProcessor<double> p(libcgen);
    
    CppAD::cg::GccCompiler<double> compiler; 
    compiler.addCompileFlag("-B/usr/lib/gcc/x86_64-linux-gnu/11/");
    auto library = p.createDynamicLibrary(compiler);

    const auto& model_lib= library->model("collision_gradient");

    // 3. Ground Truth Verification Loop
    pinocchio::Data data_gt(info.model);
    Eigen::VectorXd q = Eigen::VectorXd::Zero(nq);
    std::vector<double> input_vec(input_dim);
    std::vector<double> output_vec(nv);
    
    int n_tests = 100;
    double max_error = 0.0;
    
    // Random number generation
    std::srand(std::time(nullptr));

    fmt::print("\n--- Verifying Collision Gradient Consistency ---\n");

    for(int k=0; k<n_tests; ++k) {
        // Random q
        q = pinocchio::randomConfiguration(info.model);
        
        // Fill input vector
        for(size_t i=0; i<nq; ++i) input_vec[i] = q[i];

        // Ground Truth Computation
        pinocchio::computeJointJacobians(info.model, data_gt, q);
        pinocchio::framesForwardKinematics(info.model, data_gt, q);
        
        Eigen::VectorXd gt_grad = Eigen::VectorXd::Zero(nv);
        pinocchio::Data::Matrix6x J_temp(6, nv);

        // Random Gradients & GT Accumulation
        int grad_ptr = nq;
        auto compute_gt_sphere = [&](const SphereInfo& s) {
            Eigen::Vector3d g;
            g.setRandom(); // Random gradient
            
            // Store in input for generated code
            input_vec[grad_ptr++] = g[0];
            input_vec[grad_ptr++] = g[1];
            input_vec[grad_ptr++] = g[2];

            // GT Logic
            J_temp.setZero();
            pinocchio::getJointJacobian(info.model, data_gt, s.parent_joint, pinocchio::LOCAL_WORLD_ALIGNED, J_temp);
            
            Eigen::Vector3d offset = data_gt.oMi[s.parent_joint].rotation() * s.relative.translation();
            Eigen::Vector3d tau = offset.cross(g);
            
            pinocchio::Force f;
            f.linear() = g;
            f.angular() = tau;
            
            gt_grad += J_temp.transpose() * f.toVector();
        };

        for (const auto &s : info.spheres) compute_gt_sphere(s);
        for (auto i = 0U; i < info.model.frames.size(); ++i) {
            if (info.bounding_spheres.count(i)) compute_gt_sphere(info.bounding_spheres.at(i));
        }

        // Run Generated Code
        output_vec = model_lib->ForwardZero(input_vec);

        // Compare
        for(int i=0; i<nv; ++i) {
            double err = std::abs(output_vec[i] - gt_grad[i]);
            if(err > max_error) max_error = err;
        }
    }

    fmt::print("Verified {} tests. Max Error: {:.6e}\n", n_tests, max_error);
    if(max_error > 1e-5) {
        throw std::runtime_error("Verification FAILED: Gradient mismatch too large.");
    } else {
        fmt::print("Verification PASSED.\n");
    }

    // 4. Benchmarking
    int n_bench = 10000;
    fmt::print("\n--- Benchmarking ({} iterations) ---\n", n_bench);

    // generated
    auto start_gen = std::chrono::high_resolution_clock::now();
    for(int k=0; k<n_bench; ++k) {
         model_lib->ForwardZero(input_vec);
    }
    auto end_gen = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff_gen = end_gen - start_gen;
    
    // ground truth
    // Note: re-using last q and gradients to avoid overhead of random generation
    auto start_gt = std::chrono::high_resolution_clock::now();
    for(int k=0; k<n_bench; ++k) {
        pinocchio::computeJointJacobians(info.model, data_gt, q);
        pinocchio::framesForwardKinematics(info.model, data_gt, q);
        Eigen::VectorXd gt_grad = Eigen::VectorXd::Zero(nv);
        // ... (simplified loop for speed, assuming same structure)
         pinocchio::Data::Matrix6x J_temp(6, nv);
         // Just do one representative sphere interaction to model the loop cost? 
         // No, we must replicate the full loop for fair comparison.
         // We'll iterate all spheres.
         auto iter_spheres = [&](const SphereInfo& s) {
             J_temp.setZero();
             pinocchio::getJointJacobian(info.model, data_gt, s.parent_joint, pinocchio::LOCAL_WORLD_ALIGNED, J_temp);
             // Dummy math to simulate cost
             Eigen::Vector3d offset = data_gt.oMi[s.parent_joint].rotation() * s.relative.translation();
             // We use pre-existing input_vec values
             // (Accessing random indices to simulate read)
             Eigen::Vector3d g(0.1, 0.2, 0.3); 
             Eigen::Vector3d tau = offset.cross(g);
             gt_grad += J_temp.transpose() * (Eigen::VectorXd(6) << g, tau).finished();
         };
         for (const auto &s : info.spheres) iter_spheres(s);
         for (auto i = 0U; i < info.model.frames.size(); ++i) {
            if (info.bounding_spheres.count(i)) iter_spheres(info.bounding_spheres.at(i));
         }
    }
    auto end_gt = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff_gt = end_gt - start_gt;

    fmt::print("Generated Code Time: {:.4f} s ({:.2f} us/iter)\n", 
        diff_gen.count(), diff_gen.count() * 1e6 / n_bench);
    fmt::print("Pinocchio (GT) Time: {:.4f} s ({:.2f} us/iter)\n", 
        diff_gt.count(), diff_gt.count() * 1e6 / n_bench);
    fmt::print("Speedup: {:.2f}x\n", diff_gt.count() / diff_gen.count());
}

int main(int argc, char **argv)
{
    cxxopts::Options options(argv[0], "Tracing compiler for forward kinematics and collision checking");

    options.positional_help("[JSON configuration filename]").show_positional_help();

    options.add_options()                                                                       //
        ("f,configuration_file", "JSON configuration filename", cxxopts::value<std::string>())  //
        ("o,output_filename", "Output JSON filename", cxxopts::value<std::string>())            //
        ("t,output_template",
         "Output template filename (override configuration file)",
         cxxopts::value<std::string>())  //
        ("v,verify", "Verify consistency against Pinocchio and benchmark")
        ("h,help", "Print usage")        //
        ;

    options.parse_positional({"configuration_file"});

    auto result = options.parse(argc, argv);

    if (result.count("help"))
    {
        std::cout << options.help() << std::endl;
        exit(0);
    }

    if (not result.count("configuration_file"))
    {
        throw std::runtime_error(fmt::format("Must provide configuration file!"));
    }

    std::filesystem::path json_path(result["configuration_file"].as<std::string>());
    auto parent_path = json_path.parent_path();

    if (not std::filesystem::exists(json_path))
    {
        throw std::runtime_error(fmt::format("JSON file {} does not exist!", json_path.string()));
    }

    if (not std::filesystem::exists(json_path))
    {
    }

    std::ifstream json_file(json_path);
    nlohmann::json data;

    try
    {
        data = nlohmann::json::parse(json_file);
    }
    catch (std::exception &e)
    {
        throw std::runtime_error(fmt::format("Failed to parse JSON file! Error: \n{}", e.what()));
    }

    std::optional<std::filesystem::path> srdf_path = {};
    if (data.contains("srdf"))
    {
        srdf_path = parent_path / data["srdf"];
    }

    std::optional<std::string> end_effector_name = {};
    if (data.contains("end_effector"))
    {
        end_effector_name = data["end_effector"];
    }

    std::string language = "c++";
    if (data.contains("language"))
    {
        language = data["language"];
    }

    RobotInfo robot(parent_path / data["urdf"], srdf_path, end_effector_name);

    if (result.count("verify"))
    {
        verify_collision_gradient_correctness(robot);
        return 0; // Exit after verification
    }

    data.update(robot.json());


    auto traced_eefk_code = trace_sphere_cc_fk(robot, language, false, false, true);
    data["eefk_code"] = traced_eefk_code.code;
    data["eefk_code_vars"] = traced_eefk_code.temp_variables;
    data["eefk_code_output"] = traced_eefk_code.outputs;

    auto traced_spherefk_code = trace_sphere_cc_fk(robot, language, true, false, false);
    data["spherefk_code"] = traced_spherefk_code.code;
    data["spherefk_code_vars"] = traced_spherefk_code.temp_variables;
    data["spherefk_code_output"] = traced_spherefk_code.outputs;

    auto traced_ccfk_code = trace_sphere_cc_fk(robot, language, true, true, false);
    data["ccfk_code"] = traced_ccfk_code.code;
    data["ccfk_code_vars"] = traced_ccfk_code.temp_variables;
    data["ccfk_code_output"] = traced_ccfk_code.outputs;

    auto traced_ccfkee_code = trace_sphere_cc_fk(robot, language, true, true, true);
    data["ccfkee_code"] = traced_ccfkee_code.code;
    data["ccfkee_code_vars"] = traced_ccfkee_code.temp_variables;
    data["ccfkee_code_output"] = traced_ccfkee_code.outputs;

    auto traced_colgrad_code = trace_collision_gradient(robot, language, true, true);
    data["colgrad_code"] = traced_colgrad_code.code;
    data["colgrad_code_vars"] = traced_colgrad_code.temp_variables;
    data["colgrad_code_output"] = traced_colgrad_code.outputs;


    inja::Environment env;

    for (const auto &subt : data["subtemplates"])
    {
        inja::Template temp = env.parse_template(parent_path / subt["template"]);
        env.include_template(subt["name"], temp);
    }

    std::string output_template;
    if (result.count("output_template"))
    {
        output_template = result["output_template"].as<std::string>();
    }
    else
    {
        output_template = data["output"];
    }

    inja::Template temp = env.parse_template(parent_path / data["template"]);
    env.write(temp, data, output_template);

    std::string output_filename;
    if (result.count("output_filename"))
    {
        output_filename = result["output_filename"].as<std::string>();
    }
    else
    {
        output_filename = "output.json";
    }

    std::ofstream output_file(output_filename);
    output_file << data.dump();
    output_file.close();

    return 0;
}
