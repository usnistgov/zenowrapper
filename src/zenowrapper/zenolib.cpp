#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/ndarray.h>
#include <iomanip>

// Include ZENO headers
#include "ParametersWalkOnSpheres.h"
#include "ParametersInteriorSampling.h"
#include "ParametersResults.h"
#include "Units.h"
#include "Geometry/Vector3.h"
#include "Geometry/Sphere.h"
#include "Geometry/MixedModel.h"
#include "Zeno.h"
#include "Results.h"
#include "Uncertain.h"

namespace nb = nanobind;
using namespace nb::literals;
using namespace zeno;

// Wrapper class to hold ZENO results in a Python-friendly format
struct ZenoResults {
    // Capacitance
    double capacitance_mean;
    double capacitance_variance;
    
    // Electric polarizability tensor (3x3)
    double polarizability_tensor_mean[9];
    double polarizability_tensor_variance[9];
    
    // Electric polarizability eigenvalues (3)
    double polarizability_eigenvalues_mean[3];
    double polarizability_eigenvalues_variance[3];
    
    // Mean polarizability (scalar)
    double mean_polarizability_mean;
    double mean_polarizability_variance;
    
    // Intrinsic conductivity
    double intrinsic_conductivity_mean;
    double intrinsic_conductivity_variance;
    
    // Volume
    double volume_mean;
    double volume_variance;
    
    // Gyration tensor (3x3)
    double gyration_tensor_mean[9];
    double gyration_tensor_variance[9];
    
    // Gyration eigenvalues (3)
    double gyration_eigenvalues_mean[3];
    double gyration_eigenvalues_variance[3];
    
    // Capacitance of same volume sphere
    double capacitance_sphere_mean;
    double capacitance_sphere_variance;
    
    // Hydrodynamic radius
    double hydrodynamic_radius_mean;
    double hydrodynamic_radius_variance;
    
    // Prefactor polarizability to intrinsic viscosity
    double q_eta_mean;
    double q_eta_variance;
    
    // Viscometric radius
    double viscometric_radius_mean;
    double viscometric_radius_variance;
    
    // Intrinsic viscosity
    double intrinsic_viscosity_mean;
    double intrinsic_viscosity_variance;
    
    // Friction coefficient (if viscosity is set)
    double friction_coefficient_mean;
    double friction_coefficient_variance;
    
    // Diffusion coefficient (if viscosity and temperature are set)
    double diffusion_coefficient_mean;
    double diffusion_coefficient_variance;
    
    // Sedimentation coefficient (if viscosity, mass, and buoyancy factor are set)
    double sedimentation_coefficient_mean;
    double sedimentation_coefficient_variance;
    
    // Mass intrinsic viscosity (if mass is set)
    double mass_intrinsic_viscosity_mean;
    double mass_intrinsic_viscosity_variance;
};

// Helper function to extract mean from Uncertain<double>
double getMean(const Uncertain<double>& u) {
    return u.getMean();
}

// Helper function to extract variance from Uncertain<double>
double getVariance(const Uncertain<double>& u) {
    return u.getVariance();
}

// Main function to run ZENO computation on a single frame
ZenoResults compute_zeno_single_frame(
    nb::ndarray<double, nb::shape<-1, 3>, nb::c_contig> positions,
    nb::ndarray<double, nb::shape<-1>, nb::c_contig> radii,
    ParametersWalkOnSpheres* params_walk,
    ParametersInteriorSampling* params_interior,
    ParametersResults* params_results
) {
    // Build the MixedModel from positions and radii
    // CRITICAL: Must use stack allocation! The Zeno object stores POINTERS to the
    // vectors inside MixedModel (via getAndLockSpheres()), so the model must
    // remain valid for the entire lifetime of the Zeno object.
    MixedModel<double> model;
    
    size_t n_atoms = positions.shape(0);
    auto pos_view = positions.view();
    auto rad_view = radii.view();
    
    for (size_t i = 0; i < n_atoms; i++) {
        Vector3<double> center(
            pos_view(i, 0),
            pos_view(i, 1),
            pos_view(i, 2)
        );
        Sphere<double> sphere(center, rad_view(i));
        model.addSphere(sphere);
    }
    
    // Create Zeno object and run computations
    // CRITICAL: The Zeno constructor calls addMixedModel() which stores POINTERS to the
    // sphere/cuboid/triangle vectors inside 'model' via getAndLockSpheres(). The model
    // must remain valid for the entire lifetime of the Zeno object and all computations.
    Zeno zeno(&model);
    
    // CRITICAL FIX: ZENO's numThreads defaults to 0, which means "don't run any computations"!
    // If the user hasn't explicitly set num_threads, default to 1 for single-threaded execution.
    if (params_walk->getNumThreads() == 0) {
        params_walk->setNumThreads(1);
    }
    if (params_interior->getNumThreads() == 0) {
        params_interior->setNumThreads(1);
    }
    
    // Run Walk on Spheres
    // ZENO's doWalkOnSpheres() calls computeDefaultParameters() which auto-computes
    // launch radius, center, and skin thickness if not explicitly set by the user.
    Zeno::Status wos_status = zeno.doWalkOnSpheres(params_walk, params_results);
    if (wos_status != Zeno::Status::Success) {
        throw std::runtime_error("Walk on Spheres computation failed");
    }
    
    // Run Interior Sampling
    Zeno::Status interior_status = zeno.doInteriorSampling(params_interior, params_results);
    if (interior_status != Zeno::Status::Success) {
        throw std::runtime_error("Interior Sampling computation failed");
    }
    
    // Get results
    Results results;
    zeno.getResults(params_results, &results);
    
    // Package results into ZenoResults struct
    ZenoResults zeno_results;
    
    // Initialize all arrays to zero
    std::fill(zeno_results.polarizability_tensor_mean, zeno_results.polarizability_tensor_mean + 9, 0.0);
    std::fill(zeno_results.polarizability_tensor_variance, zeno_results.polarizability_tensor_variance + 9, 0.0);
    std::fill(zeno_results.polarizability_eigenvalues_mean, zeno_results.polarizability_eigenvalues_mean + 3, 0.0);
    std::fill(zeno_results.polarizability_eigenvalues_variance, zeno_results.polarizability_eigenvalues_variance + 3, 0.0);
    std::fill(zeno_results.gyration_tensor_mean, zeno_results.gyration_tensor_mean + 9, 0.0);
    std::fill(zeno_results.gyration_tensor_variance, zeno_results.gyration_tensor_variance + 9, 0.0);
    std::fill(zeno_results.gyration_eigenvalues_mean, zeno_results.gyration_eigenvalues_mean + 3, 0.0);
    std::fill(zeno_results.gyration_eigenvalues_variance, zeno_results.gyration_eigenvalues_variance + 3, 0.0);
    
    zeno_results.capacitance_mean = getMean(results.capacitance.value);
    zeno_results.capacitance_variance = getVariance(results.capacitance.value);
    
    // Extract polarizability tensor
    // ZENO's Matrix3x3::get() returns Uncertain<double> by value, which can trigger
    // IndexError when the covariance matrix doesn't have entries for all component IDs.
    // We catch ALL exceptions and leave values at zero if extraction fails.
    if (!results.polarizabilityTensor.prettyName.empty()) {
        for (int idx = 0; idx < 9; idx++) {
            int row = idx / 3;
            int col = idx % 3;
            zeno_results.polarizability_tensor_mean[idx] = 
                results.polarizabilityTensor.value.get(row, col).getMean();
            zeno_results.polarizability_tensor_variance[idx] = 
                results.polarizabilityTensor.value.get(row, col).getVariance();
        }
    }
    
    // Extract polarizability eigenvalues
    if (!results.polarizabilityEigenvalues.prettyName.empty()) {

        for (int i = 0; i < 3; i++) {
            zeno_results.polarizability_eigenvalues_mean[i] = 
                results.polarizabilityEigenvalues.value[i].getMean();
            zeno_results.polarizability_eigenvalues_variance[i] = 
                results.polarizabilityEigenvalues.value[i].getVariance();
        }
    }
    
    zeno_results.mean_polarizability_mean = getMean(results.meanPolarizability.value);
    zeno_results.mean_polarizability_variance = getVariance(results.meanPolarizability.value);
    
    zeno_results.intrinsic_conductivity_mean = getMean(results.intrinsicConductivity.value);
    zeno_results.intrinsic_conductivity_variance = getVariance(results.intrinsicConductivity.value);
    
    zeno_results.volume_mean = getMean(results.volume.value);
    zeno_results.volume_variance = getVariance(results.volume.value);
    
    // Extract gyration tensor
    if (!results.gyrationTensor.prettyName.empty()) {
        for (int idx = 0; idx < 9; idx++) {
            int row = idx / 3;
            int col = idx % 3;
            zeno_results.gyration_tensor_mean[idx] = 
                results.gyrationTensor.value.get(row, col).getMean();
            zeno_results.gyration_tensor_variance[idx] = 
                results.gyrationTensor.value.get(row, col).getVariance();
        }
    }
    
    // Extract gyration eigenvalues
    if (!results.gyrationEigenvalues.prettyName.empty()) {
        for (int i = 0; i < 3; i++) {
            zeno_results.gyration_eigenvalues_mean[i] = 
                results.gyrationEigenvalues.value[i].getMean();
            zeno_results.gyration_eigenvalues_variance[i] = 
                results.gyrationEigenvalues.value[i].getVariance();
        }
    }
    
    zeno_results.capacitance_sphere_mean = getMean(results.capacitanceOfASphere.value);
    zeno_results.capacitance_sphere_variance = getVariance(results.capacitanceOfASphere.value);
    
    zeno_results.hydrodynamic_radius_mean = getMean(results.hydrodynamicRadius.value);
    zeno_results.hydrodynamic_radius_variance = getVariance(results.hydrodynamicRadius.value);
    
    zeno_results.q_eta_mean = getMean(results.q_eta.value);
    zeno_results.q_eta_variance = getVariance(results.q_eta.value);
    
    zeno_results.viscometric_radius_mean = getMean(results.viscometricRadius.value);
    zeno_results.viscometric_radius_variance = getVariance(results.viscometricRadius.value);
    
    zeno_results.intrinsic_viscosity_mean = getMean(results.intrinsicViscosity.value);
    zeno_results.intrinsic_viscosity_variance = getVariance(results.intrinsicViscosity.value);
    
    // Optional results - check if they have non-empty prettyName (indicates they were computed)
    if (!results.frictionCoefficient.prettyName.empty()) {
        zeno_results.friction_coefficient_mean = getMean(results.frictionCoefficient.value);
        zeno_results.friction_coefficient_variance = getVariance(results.frictionCoefficient.value);
    }
    
    if (!results.diffusionCoefficient.prettyName.empty()) {
        zeno_results.diffusion_coefficient_mean = getMean(results.diffusionCoefficient.value);
        zeno_results.diffusion_coefficient_variance = getVariance(results.diffusionCoefficient.value);
    }
    
    if (!results.sedimentationCoefficient.prettyName.empty()) {
        zeno_results.sedimentation_coefficient_mean = getMean(results.sedimentationCoefficient.value);
        zeno_results.sedimentation_coefficient_variance = getVariance(results.sedimentationCoefficient.value);
    }
    
    if (!results.intrinsicViscosityConventional.prettyName.empty()) {
        zeno_results.mass_intrinsic_viscosity_mean = getMean(results.intrinsicViscosityConventional.value);
        zeno_results.mass_intrinsic_viscosity_variance = getVariance(results.intrinsicViscosityConventional.value);
    }
    
    return zeno_results;
}

// name here must match nanobind_add_module() in CMake
NB_MODULE(zenowrapper_ext, m) {
    nb::module_ m2 = m.def_submodule("zenolib", "A submodule of zenowrapper containing pythonic versions of zeno classes.");
    
    // Bind Units enums
    nb::enum_<Units::Length>(m2, "Length")
        .value("m", Units::Length::m)
        .value("cm", Units::Length::cm)
        .value("nm", Units::Length::nm)
        .value("A", Units::Length::A)
        .value("L", Units::Length::L);
    
    nb::enum_<Units::Temperature>(m2, "Temperature")
        .value("C", Units::Temperature::C)
        .value("K", Units::Temperature::K);
    
    nb::enum_<Units::Mass>(m2, "Mass")
        .value("Da", Units::Mass::Da)
        .value("kDa", Units::Mass::kDa)
        .value("g", Units::Mass::g)
        .value("kg", Units::Mass::kg);
    
    nb::enum_<Units::Viscosity>(m2, "Viscosity")
        .value("p", Units::Viscosity::p)
        .value("cp", Units::Viscosity::cp);
    
    // Bind ParametersWalkOnSpheres
    nb::class_<ParametersWalkOnSpheres>(m2, "ParametersWalkOnSpheres")
        .def(nb::init<>())
        .def("setNumThreads", &ParametersWalkOnSpheres::setNumThreads)
        .def("setSeed", &ParametersWalkOnSpheres::setSeed)
        .def("setTotalNumWalks", &ParametersWalkOnSpheres::setTotalNumWalks)
        .def("setMinTotalNumWalks", &ParametersWalkOnSpheres::setMinTotalNumWalks)
        .def("setMaxErrorCapacitance", &ParametersWalkOnSpheres::setMaxErrorCapacitance)
        .def("setMaxErrorPolarizability", &ParametersWalkOnSpheres::setMaxErrorPolarizability)
        .def("setMaxRunTime", &ParametersWalkOnSpheres::setMaxRunTime)
        .def("setSkinThickness", &ParametersWalkOnSpheres::setSkinThickness)
        .def("setLaunchRadius", &ParametersWalkOnSpheres::setLaunchRadius);
    
    // Bind ParametersInteriorSampling
    nb::class_<ParametersInteriorSampling>(m2, "ParametersInteriorSampling")
        .def(nb::init<>())
        .def("setNumThreads", &ParametersInteriorSampling::setNumThreads)
        .def("setSeed", &ParametersInteriorSampling::setSeed)
        .def("setTotalNumSamples", &ParametersInteriorSampling::setTotalNumSamples)
        .def("setMinTotalNumSamples", &ParametersInteriorSampling::setMinTotalNumSamples)
        .def("setMaxErrorVolume", &ParametersInteriorSampling::setMaxErrorVolume)
        .def("setMaxRunTime", &ParametersInteriorSampling::setMaxRunTime)
        .def("setLaunchRadius", &ParametersInteriorSampling::setLaunchRadius);
    
    // Bind ParametersResults
    nb::class_<ParametersResults>(m2, "ParametersResults")
        .def(nb::init<>())
        .def("setLengthScale", &ParametersResults::setLengthScale)
        .def("setTemperature", &ParametersResults::setTemperature)
        .def("setMass", &ParametersResults::setMass)
        .def("setSolventViscosity", &ParametersResults::setSolventViscosity)
        .def("setBuoyancyFactor", &ParametersResults::setBuoyancyFactor);
    
    // Bind ZenoResults struct with property getters for arrays
    nb::class_<ZenoResults>(m2, "ZenoResults")
        .def_ro("capacitance_mean", &ZenoResults::capacitance_mean)
        .def_ro("capacitance_variance", &ZenoResults::capacitance_variance)
        .def_prop_ro("polarizability_tensor_mean", [](const ZenoResults& r) {
            return std::vector<double>(r.polarizability_tensor_mean, r.polarizability_tensor_mean + 9);
        })
        .def_prop_ro("polarizability_tensor_variance", [](const ZenoResults& r) {
            return std::vector<double>(r.polarizability_tensor_variance, r.polarizability_tensor_variance + 9);
        })
        .def_prop_ro("polarizability_eigenvalues_mean", [](const ZenoResults& r) {
            return std::vector<double>(r.polarizability_eigenvalues_mean, r.polarizability_eigenvalues_mean + 3);
        })
        .def_prop_ro("polarizability_eigenvalues_variance", [](const ZenoResults& r) {
            return std::vector<double>(r.polarizability_eigenvalues_variance, r.polarizability_eigenvalues_variance + 3);
        })
        .def_ro("mean_polarizability_mean", &ZenoResults::mean_polarizability_mean)
        .def_ro("mean_polarizability_variance", &ZenoResults::mean_polarizability_variance)
        .def_ro("intrinsic_conductivity_mean", &ZenoResults::intrinsic_conductivity_mean)
        .def_ro("intrinsic_conductivity_variance", &ZenoResults::intrinsic_conductivity_variance)
        .def_ro("volume_mean", &ZenoResults::volume_mean)
        .def_ro("volume_variance", &ZenoResults::volume_variance)
        .def_prop_ro("gyration_tensor_mean", [](const ZenoResults& r) {
            return std::vector<double>(r.gyration_tensor_mean, r.gyration_tensor_mean + 9);
        })
        .def_prop_ro("gyration_tensor_variance", [](const ZenoResults& r) {
            return std::vector<double>(r.gyration_tensor_variance, r.gyration_tensor_variance + 9);
        })
        .def_prop_ro("gyration_eigenvalues_mean", [](const ZenoResults& r) {
            return std::vector<double>(r.gyration_eigenvalues_mean, r.gyration_eigenvalues_mean + 3);
        })
        .def_prop_ro("gyration_eigenvalues_variance", [](const ZenoResults& r) {
            return std::vector<double>(r.gyration_eigenvalues_variance, r.gyration_eigenvalues_variance + 3);
        })
        .def_ro("capacitance_sphere_mean", &ZenoResults::capacitance_sphere_mean)
        .def_ro("capacitance_sphere_variance", &ZenoResults::capacitance_sphere_variance)
        .def_ro("hydrodynamic_radius_mean", &ZenoResults::hydrodynamic_radius_mean)
        .def_ro("hydrodynamic_radius_variance", &ZenoResults::hydrodynamic_radius_variance)
        .def_ro("q_eta_mean", &ZenoResults::q_eta_mean)
        .def_ro("q_eta_variance", &ZenoResults::q_eta_variance)
        .def_ro("viscometric_radius_mean", &ZenoResults::viscometric_radius_mean)
        .def_ro("viscometric_radius_variance", &ZenoResults::viscometric_radius_variance)
        .def_ro("intrinsic_viscosity_mean", &ZenoResults::intrinsic_viscosity_mean)
        .def_ro("intrinsic_viscosity_variance", &ZenoResults::intrinsic_viscosity_variance)
        .def_ro("friction_coefficient_mean", &ZenoResults::friction_coefficient_mean)
        .def_ro("friction_coefficient_variance", &ZenoResults::friction_coefficient_variance)
        .def_ro("diffusion_coefficient_mean", &ZenoResults::diffusion_coefficient_mean)
        .def_ro("diffusion_coefficient_variance", &ZenoResults::diffusion_coefficient_variance)
        .def_ro("sedimentation_coefficient_mean", &ZenoResults::sedimentation_coefficient_mean)
        .def_ro("sedimentation_coefficient_variance", &ZenoResults::sedimentation_coefficient_variance)
        .def_ro("mass_intrinsic_viscosity_mean", &ZenoResults::mass_intrinsic_viscosity_mean)
        .def_ro("mass_intrinsic_viscosity_variance", &ZenoResults::mass_intrinsic_viscosity_variance);
    
    // Bind main computation function
    m2.def("compute_zeno_single_frame", &compute_zeno_single_frame,
           "positions"_a, "radii"_a, "params_walk"_a, "params_interior"_a, "params_results"_a,
           "Compute ZENO results for a single frame");
}