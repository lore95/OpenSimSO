# # Get a handle to the current model and create a new copy 
import opensim as osim
import os, os.path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import xml.etree.ElementTree as ET


plt.rcParams.update({'font.size': 20}) 
def createPAndDFlexionXml(dir_path):
    # Define directory path

    # Define muscle groups
    Pmuscles = ['tib_ant_r', 'ext_dig_r', 'ext_hal_r', 'per_tert_r']
    Dmuscles = ['med_gas_r', 'soleus_r', 'tib_post_r', 'per_brev_r', 'per_long_r']

    # Create two DataFrames for the output Excel files
    dorsiflexion_data = pd.DataFrame(columns=["stiffness", "activation"])
    plantarflexion_data = pd.DataFrame(columns=["stiffness", "activation"])

    # Loop through the directory
    for root, dirs, files in os.walk(dir_path):
        for file_name in files:
            # Check if the file is a `.sto` file and ends with "activation"
            if file_name.endswith("activation.sto"):
                file_path = os.path.join(root, file_name)  # Full path of the file

                try:
                    # Read the `.sto` file, skipping the header rows
                    ACT = pd.read_csv(file_path, delim_whitespace=True, skiprows=8)
                    
                    # Extract stiffness value from the filename
                    # Assuming filenames are in the format "SO_<stiffness>_something.activation.sto"
                    try:
                        stiff = int(file_name.split('_')[1])  # Extract stiffness as an integer
                        print(stiff)
                    except ValueError:
                        print(f"Unable to extract stiffness from file name: {file_name}")
                        continue

                    # Check if the required columns exist in the data
                    if all(muscle in ACT.columns for muscle in Pmuscles + Dmuscles):
                        # Calculate the average activation for Pmuscles and Dmuscles
                        avg_Pmuscles = ACT[Pmuscles].mean().mean()
                        avg_Dmuscles = ACT[Dmuscles].mean().mean()
                        
                        # Add data to the respective DataFrames
                        dorsiflexion_data = pd.concat([dorsiflexion_data, 
                                                    pd.DataFrame({"stiffness": [stiff], "activation": [avg_Dmuscles]})])
                        plantarflexion_data = pd.concat([plantarflexion_data, 
                                                        pd.DataFrame({"stiffness": [stiff], "activation": [avg_Pmuscles]})])
                    else:
                        print(f"File {file_name} is missing required muscle columns.")
                except Exception as e:
                    print(f"Error reading file {file_name}: {e}")
    # Save the results to Excel files
    
    dorsiflexion_data = dorsiflexion_data.sort_values(by='stiffness', ascending=True)
    dorsiflexion_data.reset_index(drop=True, inplace=True)
    dorsiflexion_data.to_excel("Solutions/ActivationsExls/DorsiflexionOverStiffness.xlsx", index=False)

    plantarflexion_data = plantarflexion_data.sort_values(by='stiffness', ascending=True)
    plantarflexion_data.reset_index(drop=True, inplace=True)
    plantarflexion_data.to_excel("Solutions/ActivationsExls/PlantarflexionOverStiffness.xlsx", index=False)
    
    merged_data = pd.merge(
        dorsiflexion_data,
        plantarflexion_data,
        on='stiffness',
        suffixes=('_dorsi', '_planta')
    )

    # Calculate the sum of activations
    merged_data['activation_sum'] = (
        merged_data['activation_dorsi'] + merged_data['activation_planta']
    )

    
    plt.figure(figsize=(10, 6))

    # Plot the Sum of Activations
    plt.plot(
        merged_data['stiffness'],
        merged_data['activation_sum'],
        label="Sum of Activations",
        color='red',
        linestyle='--',
        marker='s'
    )
    # Plot Dorsiflexion data
    plt.plot(
        dorsiflexion_data['stiffness'],
        dorsiflexion_data['activation'],
        label="Dorsiflexion",
        color='blue',
        marker='o'
    )

    # Plot Plantarflexion data
    plt.plot(
        plantarflexion_data['stiffness'],
        plantarflexion_data['activation'],
        label="Plantarflexion",
        color='green',
        marker='x'
    )

    # Add title and labels
    plt.title("Activation vs Stiffness")
    plt.xlabel("Stiffness")
    plt.ylabel("Activation")

    # Add grid, legend, and adjust layout
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # Save and display the plot
    plt.savefig("Activation_vs_Stiffness_Combined.png")
    plt.show()

def calculateAverageActivation(dir_path):
    # Initialize variables to track the file with minimum activation
    min_activation = float('inf')
    min_file = None

    # Iterate over all directories and subdirectories in the given directory
    for root, dirs, files in os.walk(dir_path):  # Use os.walk() to traverse subdirectories
        for file_name in files:
            # Check if the file is a `.sto` file and ends with "activation"
            if file_name.endswith('activation.sto'):
                file_path = os.path.join(root, file_name)  # Full path of the file

                try:
                    # Read the file
                    ACT = pd.read_csv(file_path, delim_whitespace=True, skiprows=8)
                    
                    # Extract activations for the muscles
                    Pmuscles = ['tib_ant_r', 'ext_dig_r', 'ext_hal_r', 'per_tert_r']
                    Dmuscles = ['med_gas_r', 'soleus_r', 'tib_post_r', 'per_brev_r', 'per_long_r']
                    
                    #     'rect_fem_r', 'vas_lat_r', 'vas_int_r', 'semiten_r', 'bifemlh_r',
                    #     'iliacus_r', 'psoas_r', 'sar_r', 'glut_max1_r', 'glut_max2_r',
                    #     'glut_max3_r', 'bifemsh_r', 'semimem_r', 'grac_r'
                    # ]
                    
                    # Ensure all required columns exist in the file
                    if all(muscle in ACT.columns for muscle in muscles):
                        # Sum activations for the current file
                        total_activation = ACT[muscles].sum().sum()
                        
                        # Update minimum activation and corresponding file
                        if total_activation < min_activation:
                            min_activation = total_activation
                            min_file = file_path  # Store full path of the file
                    else:
                        print(f"File {file_name} is missing required muscle columns.")
                
                except Exception as e:
                    print(f"Error reading file {file_name}: {e}")

    # Return the file with minimum activation and the value
    for i in range(50):
        print("****************************************")
    print(f"File with minimum activation: {min_file}")
    print(f"Minimum activation value: {min_activation}")
    return min_file, min_activation


# def calculateAverageActivation(file_path):
#     #dorsiflexors
#     ACT = pd.read_csv(file_path, delim_whitespace=True, skiprows=8)  # Adjust the header skip count if necessary
#     tib_ant_r = ACT['tib_ant_r']
#     ext_dig_r = ACT['ext_dig_r']
#     ext_hal_r = ACT['ext_hal_r']
#     per_tert_r = ACT['per_tert_r']
    
#     #plantarflexors
#     med_gas_r = ACT['med_gas_r']
#     soleus_r = ACT['soleus_r']
#     tib_post_r = ACT['tib_post_r']
#     per_brev_r = ACT['per_brev_r']
#     per_long_r = ACT['per_long_r']

#     #extensors
def plotAllInDir():
    file_dir = "Solutions/SO"
    for file_name in os.listdir(file_dir):
        # Check if the file ends with 'activation.sto'
        if file_name.endswith('activation.sto'):
            # Split the file name to extract stiffness
            try:
                stiffness = file_name.split('_')[1]
                output_path = "Solutions/plots/"+stiffness
                print(file_name)
                os.makedirs(output_path, exist_ok=True)  # exist_ok=True prevents errors if the directory already exists
                file_name = "Solutions/SO/" + file_name
                plotDorsiFlection(file_name,output_path,stiffness)
                plotPlantarFlection(file_name,output_path,stiffness)
                plotKneeExtension(file_name,output_path,stiffness)
                plotKneeFlection(file_name,output_path,stiffness)
                plotHipExtension(file_name,output_path,stiffness)
                plotHipFlection(file_name,output_path,stiffness)
            except ValueError:
                print(f"Error extracting stiffness from file: {file_name}")

def plotAll(file_path,stiffness):
    output_path = "Solutions/plots/"+stiffness

    os.makedirs(output_path, exist_ok=True)  # exist_ok=True prevents errors if the directory already exists
    plotDorsiFlection(file_path,output_path,stiffness)
    plotPlantarFlection(file_path,output_path,stiffness)
    plotKneeExtension(file_path,output_path,stiffness)
    plotKneeFlection(file_path,output_path,stiffness)
    plotHipExtension(file_path,output_path,stiffness)
    plotHipFlection(file_path,output_path,stiffness)
    
def changeXml(file_path, new_value):
    tree = ET.parse(file_path)
    root = tree.getroot()
    model_file_tag = root.find(".//model_file")
    if model_file_tag is not None:
        model_file_tag.text = new_value
    else:
        raise ValueError("The 'model_file' tag was not found in the XML.")
      # Save the changes back to the same file
    tree.write(file_path, encoding="unicode", xml_declaration=True)


# def plotTotalPlantarVsDorsi(plantar,dorsi,stiffness):
#     indices = np.arange(len(plantar))
#     gaitCycle = indices / len(plantar) * 100

#     plt.figure(figsize=(10, 8))
#     plt.plot(gaitCycle,plantar, color="r", label = "total plantarflexion")
#     plt.plot(gaitCycle,dorsi, color="b", label = "total dorsiflexion ")
#     plt.title("Plantar and Dorsi Flexion for " +stiffness+ " stiffness")
#     plt.xlabel("Gait cylce %")
#     plt.ylabel("Activation")
#     # Set the background color and display the plots
#     plt.tight_layout()
#     plt.legend()
#     plt.savefig("Solutions/plots/PlantarFlexionFor"+stiffness+"Stiffness", format='png', dpi=300)


def plotSpringForce(file_name,stiffness)
    home_dir = "Solutions/SO"
    ACT = pd.read_csv(file_path, delim_whitespace=True, skiprows=14)  # Adjust the header skip count if necessary
    BiarticularSpringDamper_tension = ACT["BiarticularSpringDamper_tension"]
    
    indices = np.arange(len(tib_ant_r))
    gaitCycle = indices / len(tib_ant_r) * 100

    plt.figure(figsize=(10, 8))
    plt.plot(gaitCycle,BiarticularSpringDamper_tension, color="r", label = "BiarticularSpringDamper_tension")
    plt.title("Plot Main Dorsiflexors for " +stiffness+ " stiffness")
    plt.xlabel("Gayte cycle %")
    plt.ylabel("Activation")
    plt.tight_layout()
    plt.legend()

    plt.savefig(output_path+"/Dorsiflexors"+stiffness+"Stiffness.png", format='png', dpi=300)


def plotDorsiFlection(file_path,output_path,stiffness):
    ACT = pd.read_csv(file_path, delim_whitespace=True, skiprows=8)  # Adjust the header skip count if necessary
    tib_ant_r = ACT['tib_ant_r']
    ext_dig_r = ACT['ext_dig_r']
    ext_hal_r = ACT['ext_hal_r']
    per_tert_r = ACT['per_tert_r']

    indices = np.arange(len(tib_ant_r))
    gaitCycle = indices / len(tib_ant_r) * 100

    plt.figure(figsize=(10, 8))
    plt.plot(gaitCycle,tib_ant_r, color="r", label = "tib_ant_r")
    plt.plot(gaitCycle,ext_dig_r, color="b", label = "ext_dig_r")
    plt.plot(gaitCycle,ext_hal_r, color="y", label = "ext_hal_r")
    plt.plot(gaitCycle,per_tert_r, color="g", label = "per_tert_r")
    plt.title("Plot Main Dorsiflexors for " +stiffness+ " stiffness")
    plt.xlabel("Gayte cycle %")
    plt.ylabel("Activation")
    plt.tight_layout()
    plt.legend()

    plt.savefig(output_path+"/Dorsiflexors"+stiffness+"Stiffness.png", format='png', dpi=300)


def plotPlantarFlection(file_path,output_path,stiffness):
    print(file_path)
    ACT = pd.read_csv(file_path, delim_whitespace=True, skiprows=8)  # Adjust the header skip count if necessary
    med_gas_r = ACT['med_gas_r']
    soleus_r = ACT['soleus_r']
    tib_post_r = ACT['tib_post_r']
    per_brev_r = ACT['per_brev_r']
    per_long_r = ACT['per_long_r']

    indices = np.arange(len(med_gas_r))
    gaitCycle = indices / len(med_gas_r) * 100

    plt.figure(figsize=(10, 8))
    plt.plot(med_gas_r, color="r", label = "med_gas_r")
    plt.plot(soleus_r, color="b", label = "soleus_r")
    plt.plot(tib_post_r, color="y", label = "tib_post_r")
    plt.plot(per_brev_r, color="g", label = "per_brev_r")
    plt.plot(per_long_r, color="black", label = "per_long_r")
    plt.title("Plot Main Plantarflexors on " +stiffness+ " stiffness")
    plt.xlabel("Gayte cycle %")
    plt.ylabel("Activation")
    plt.tight_layout()
    plt.legend()

    plt.savefig(output_path+"/Plantarflexors"+stiffness+"Stiffness.png", format='png', dpi=300)


def plotKneeFlection(file_path,output_path,stiffness):
    print(file_path)
    ACT = pd.read_csv(file_path, delim_whitespace=True, skiprows=8)  # Adjust the header skip count if necessary
    
    semiten_r = ACT['semiten_r']
    bifemlh_r = ACT['bifemlh_r']
    sar_r = ACT['sar_r']
    grac_r = ACT['grac_r']

    indices = np.arange(len(bifemlh_r))
    gaitCycle = indices / len(bifemlh_r) * 100

    plt.figure(figsize=(10, 8))
    plt.plot(gaitCycle,semiten_r, color="r", label = "semiten_r")
    plt.plot(gaitCycle,bifemlh_r, color="b", label = "bifemlh_r")
    plt.plot(gaitCycle,sar_r, color="y", label = "sar_r")
    plt.plot(gaitCycle,grac_r, color="g", label = "grac_r")
    plt.title("Plot Main Knee Flexors on " +stiffness+ " stiffness")
    plt.xlabel("Gayte cycle %")
    plt.ylabel("Activation")
    plt.tight_layout()
    plt.legend()

    plt.savefig(output_path+"/KneeFlexionFor"+stiffness+"Stiffness.png", format='png', dpi=300)

def plotKneeExtension(file_path,output_path,stiffness):
    print(file_path)
    ACT = pd.read_csv(file_path, delim_whitespace=True, skiprows=8)  # Adjust the header skip count if necessary
    rect_fem_r = ACT['rect_fem_r']
    vas_lat_r = ACT['vas_lat_r']
    vas_int_r = ACT['vas_int_r']


    indices = np.arange(len(rect_fem_r))
    gaitCycle = indices / len(rect_fem_r) * 100

    plt.figure(figsize=(10, 8))
    plt.plot(rect_fem_r, color="r", label = "rect_fem_r")
    plt.plot(vas_lat_r, color="b", label = "vas_lat_r")
    plt.plot(vas_int_r, color="y", label = "vas_int_r")
    plt.title("Plot Main Knee Extensors on " +stiffness+ " stiffness")
    plt.xlabel("Gayte cycle %")
    plt.ylabel("Activation")
    plt.tight_layout()
    plt.legend()

    plt.savefig(output_path+"/KneeExtensionFor"+stiffness+"Stiffness.png", format='png', dpi=300)

def plotHipFlection(file_path,output_path,stiffness):
    print(file_path)
    ACT = pd.read_csv(file_path, delim_whitespace=True, skiprows=8)  # Adjust the header skip count if necessary
    rect_fem_r = ACT['rect_fem_r']
    iliacus_r = ACT['iliacus_r']
    psoas_r = ACT['psoas_r']
    sar_r = ACT['sar_r']

    indices = np.arange(len(rect_fem_r))
    gaitCycle = indices / len(rect_fem_r) * 100

    plt.figure(figsize=(10, 8))
    plt.plot(rect_fem_r, color="r", label = "rect_fem_r")
    plt.plot(iliacus_r, color="b", label = "iliacus_r")
    plt.plot(psoas_r, color="y", label = "psoas_r")
    plt.plot(sar_r, color="g", label = "sar_r")
    plt.title("Plot Main Hip Flexors on " +stiffness+ " stiffness")
    plt.xlabel("Gayte cycle %")
    plt.ylabel("Activation")
    plt.tight_layout()
    plt.legend()

    plt.savefig(output_path+"/HipFlexionFor"+stiffness+"Stiffness.png", format='png', dpi=300)

def plotHipExtension(file_path,output_path,stiffness):
    print(file_path)
    ACT = pd.read_csv(file_path, delim_whitespace=True, skiprows=8)  # Adjust the header skip count if necessary
    glut_max1_r = ACT['glut_max1_r']
    glut_max2_r = ACT['glut_max2_r']
    glut_max3_r = ACT['glut_max3_r']
    bifemsh_r = ACT['bifemsh_r']
    semimem_r = ACT['semimem_r']
    grac_r = ACT['grac_r']
    

    indices = np.arange(len(glut_max1_r))
    gaitCycle = indices / len(glut_max1_r) * 100

    plt.figure(figsize=(10, 8))
    plt.plot(glut_max1_r, color="r", label = "glut_max1_r")
    plt.plot(glut_max2_r, color="b", label = "glut_max2_r")
    plt.plot(glut_max3_r, color="y", label = "glut_max3_r")
    plt.plot(bifemsh_r, color="g", label = "bifemsh_r")
    plt.plot(semimem_r, color="black", label = "semimem_r")
    plt.plot(grac_r, color="pink", label = "grac_r")

    plt.title("Plot Main Hip Extensors for " +stiffness+ " stiffness")
    plt.xlabel("Gayte cycle %")
    plt.ylabel("Activation")
    # Set the background color and display the plots
    plt.tight_layout()
    plt.legend()

    plt.savefig(output_path+"/HipExtensionFor"+stiffness+"Stiffness.png", format='png', dpi=300)


def perform_so(model_file, ik_file, grf_file, grf_xml, reserve_actuators,
               results_dir,value,isLen):

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
    if(isLen):
        changeXml("SO_setup_walkingNormal.xml", "Solutions/Models/subject_scaled_RRA2_len_"+value+".osim")
    else:
        changeXml("SO_setup_walkingNormal.xml", "Solutions/Models/subject_scaled_RRA2_stiff_"+value+".osim")
    
    analysis = osim.AnalyzeTool("SO_setup_walkingNormal.xml")
    if (isLen):
        analysis.setName("SO_len_" + value )
        so_activations_file = "Solutions/SO/Len/SO_len_"+value+"_StaticOptimization_activation.sto"
    else:
        analysis.setName("SO_" + value )
        so_activations_file = "Solutions/SO/SO_"+value+"_StaticOptimization_activation.sto"

    analysis.setInitialTime(motion.getFirstTime())
    analysis.setFinalTime(motion.getLastTime())
    analysis.setLowpassCutoffFrequency(6)
    analysis.setCoordinatesFileName(ik_file) 
    analysis.setExternalLoadsFileName(results_dir + name + '.xml')
    analysis.setLoadModelAndInput(True)
    analysis.setResultsDir(results_dir)
    analysis.run()

    plotAll(so_activations_file,value)





def Part1(startingValue,endingValue,step):
    # Set geometry path for the path spring to match the gastrocnemius muscle
    i = startingValue
    while i < endingValue:
    
        #Load Model
        newbaseModel = osim.Model("Model/subject_scaled_RRA2.osim")
        pathSpringModel = newbaseModel.clone()
        pathSpringModel.setName(newbaseModel.getName()+'_path_spring')

        # Create the spring we'll add to the model (a PathSpring in OpenSim)
        name = 'BiarticularSpringDamper'
        restLength = 0.327
        dissipation = 0.01
        stiffness = i
        pathSpring = osim.PathSpring(name,restLength,stiffness,dissipation)

        gastroc = pathSpringModel.getMuscles().get('soleus_r')
        pathSpring.set_GeometryPath(gastroc.getGeometryPath())

        # Add the spring to the model
        pathSpringModel.addForce(pathSpring)

        # Load the model in the GUI
        newModel = osim.Model(pathSpringModel)
        # Save the model to file
        replaced_text = str(int(stiffness)).replace('.', '_')


        fullPathName= "Solutions/Models/subject_scaled_RRA2_stiff_" +  replaced_text + ".osim"
        pathSpringModel.printToXML(fullPathName)

        perform_so(fullPathName,"RRA/subject_scaled_2392_RRA_states.sto", "ExperimentalData/walking_GRF.mot", "GRF_file_walkingNormal.xml","gait2392_CMC_Actuators.xml","/Users/lorewnzo/Documents/Kth/Assignments/BioMech/Project3/OpenSimSO/Solutions/SO",replaced_text,False)
        i += step


def Part2(startingValue,endingValue,step):
    # Set geometry path for the path spring to match the gastrocnemius muscle
    i = startingValue
    while i < endingValue:
    
        #Load Model
        newbaseModel = osim.Model("Model/subject_scaled_RRA2.osim")
        pathSpringModel = newbaseModel.clone()
        pathSpringModel.setName(newbaseModel.getName()+'_path_spring')

        # Create the spring we'll add to the model (a PathSpring in OpenSim)
        name = 'BiarticularSpringDamper'
        restLength = i
        dissipation = 0.01
        stiffness = 61000
        pathSpring = osim.PathSpring(name,restLength,stiffness,dissipation)

        gastroc = pathSpringModel.getMuscles().get('soleus_r')
        pathSpring.set_GeometryPath(gastroc.getGeometryPath())

        # Add the spring to the model
        pathSpringModel.addForce(pathSpring)

        # Load the model in the GUI
        newModel = osim.Model(pathSpringModel)
        # Save the model to file
        replaced_text = str(float(restLength)).replace('.', '_')

        fullPathName= "Solutions/Models/subject_scaled_RRA2_len_" + replaced_text + ".osim"
        pathSpringModel.printToXML(fullPathName)
        

        perform_so(fullPathName,"RRA/subject_scaled_2392_RRA_states.sto", "ExperimentalData/walking_GRF.mot", "GRF_file_walkingNormal.xml","gait2392_CMC_Actuators.xml","/Users/lorewnzo/Documents/Kth/Assignments/BioMech/Project3/OpenSimSO/Solutions/SO/Len",replaced_text,True)
        i += step





# startingValue = input("What starting stiffness value do you want to use?")
# endingValue = input("What ending stiffness value do you want to use?")
# step = input("Which step for the stiffness would you like to use?")

#Part1(int(startingValue),int(endingValue),float(step))
#Part2(float(startingValue),float(endingValue),float(step))
#plotAll("Solutions/SO/","0_32")
plotAllInDir()
#calculateAverageActivation("Solutions/SO/Len")
#createPAndDFlexionXml("Solutions/SO")
