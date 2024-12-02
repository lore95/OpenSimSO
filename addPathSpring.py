# # Get a handle to the current model and create a new copy 
import opensim as osim
import os, os.path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np



def plotForMe(file_path,stiffness):
    print(file_path)
    ACT = pd.read_csv(file_path, delim_whitespace=True, skiprows=8)  # Adjust the header skip count if necessary
    tib_ant_r = ACT['tib_ant_r']
    ext_dig_r = ACT['ext_dig_r']
    ext_hal_r = ACT['ext_hal_r']
    per_tert_r = ACT['per_tert_r']

    plt.figure(figsize=(10, 8))
    plt.plot(tib_ant_r, color="r", label = "tib_ant_r")
    plt.plot(ext_dig_r, color="b", label = "ext_dig_r")
    plt.plot(ext_hal_r, color="y", label = "ext_hal_r")
    plt.plot(per_tert_r, color="g", label = "per_tert_r")
    plt.title("Plot for " +stiffness+ " stiffness")
    plt.xlabel("Index")
    plt.ylabel("Value")
    # Set the background color and display the plots
    plt.tight_layout()
    plt.show()


def perform_so(model_file, ik_file, grf_file, grf_xml, reserve_actuators,
               results_dir,stiffness):

   # model
    model = osim.Model(model_file)

    # prepare external forces xml file
    name = os.path.basename(grf_file)[:-8]
    external_loads = osim.ExternalLoads(grf_xml, True)
    # external_loads.setExternalLoadsModelKinematicsFileName(ik_file) 
    external_loads.setDataFileName(grf_file)
    # external_loads.setLowpassCutoffFrequencyForLoadKinematics(6)
    external_loads.printToXML(results_dir + name + '.xml')

    # add reserve actuators
    force_set = osim.SetForces(reserve_actuators, True)
    force_set.setMemoryOwner(False)  # model will be the owner
    for i in range(0, force_set.getSize()):
        model.updForceSet().append(force_set.get(i))

    # construct static optimization
    motion = osim.Storage(ik_file)
    static_optimization = osim.StaticOptimization()
    static_optimization.setStartTime(motion.getFirstTime())
    static_optimization.setEndTime(motion.getLastTime())
    static_optimization.setUseModelForceSet(True)
    static_optimization.setUseMusclePhysiology(True)
    static_optimization.setActivationExponent(2)
    static_optimization.setConvergenceCriterion(0.0001)
    static_optimization.setMaxIterations(100)

    model.addAnalysis(static_optimization)

    # analysis

    analysis = osim.AnalyzeTool("SO_setup_walkingNormal.xml")
    analysis.setName("SO_" + stiffness )
    analysis.setInitialTime(motion.getFirstTime())
    analysis.setFinalTime(motion.getLastTime())
    analysis.setLowpassCutoffFrequency(6)
    analysis.setCoordinatesFileName(ik_file) 
    analysis.setExternalLoadsFileName(results_dir + name + '.xml')
    analysis.setLoadModelAndInput(True)
    analysis.setResultsDir(results_dir)
    analysis.run()
    so_force_file = results_dir + name + '_so_forces.sto'
    so_activations_file = "Solutions/SO/SO_"+stiffness+"_StaticOptimization_activation.sto"
    plotForMe(so_activations_file,stiffness)



#Load Model

baseModel = osim.Model("Model/subject_scaled_RRA2.osim")
pathSpringModel = baseModel.clone()
pathSpringModel.setName(baseModel.getName()+'_path_spring')


# Create the spring we'll add to the model (a PathSpring in OpenSim)
name = 'BiarticularSpringDamper'
restLength = 0.302
dissipation = 0.01

# Set geometry path for the path spring to match the gastrocnemius muscle
for i in range(1000, 10001, 1000):  # Start at 1000, end at 10000, step by 1000
    stiffness = i
    pathSpring = osim.PathSpring(name,restLength,stiffness,dissipation)

    gastroc = pathSpringModel.getMuscles().get('soleus_r')
    pathSpring.set_GeometryPath(gastroc.getGeometryPath())

    # Add the spring to the model
    pathSpringModel.addForce(pathSpring)

    # Load the model in the GUI
    osim.Model(pathSpringModel)
    # Save the model to file
    fullPathName= "Solutions/subject_scaled_RRA2_stiff_" + str(int(stiffness)) + ".osim"
    pathSpringModel.printToXML(fullPathName)

    perform_so(fullPathName,"RRA/subject_scaled_2392_RRA_states.sto", "ExperimentalData/walking_GRF.mot", "GRF_file_walkingNormal.xml","gait2392_CMC_Actuators.xml","/Users/lorewnzo/Downloads/Project3files/Solutions/SO",str(int(stiffness)))
