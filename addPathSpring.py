# # Get a handle to the current model and create a new copy 
import opensim as osim
import os, os.path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import xml.etree.ElementTree as ET
import matplotlib.colors as mcolors

plt.rcParams.update({'font.size': 20}) 


def create_color_gradient(n_colors, start_color=(1.0, 0.0, 0.0), end_color=(0.0, 0.0, 1.0)):
    start_color_arr = np.array(start_color)
    end_color_arr = np.array(end_color)
    return np.linspace(start_color_arr, end_color_arr, n_colors)

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
                        strstiff = file_name.split('_')[3]  # Extract stiffness as an integer
                        stiff = (int(strstiff))

                    except ValueError:
                        print(f"Unable to extract stiffness from file name: {file_name}")
                        continue

                    # Check if the required columns exist in the data
                    if all(muscle in ACT.columns for muscle in Pmuscles + Dmuscles):
                        # Calculate the average activation for Pmuscles and Dmuscles
                        avg_Pmuscles = ACT[Pmuscles].mean().mean()
                        avg_Dmuscles = ACT[Dmuscles].mean().mean()
                        print(avg_Pmuscles)
                        print(avg_Dmuscles)
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
    dorsiflexion_data.to_excel("Solutions/ActivationsExlsLen/DorsiflexionOverStiffness.xlsx", index=False)

    plantarflexion_data = plantarflexion_data.sort_values(by='stiffness', ascending=True)
    plantarflexion_data.reset_index(drop=True, inplace=True)
    plantarflexion_data.to_excel("Solutions/ActivationsExlsLen/PlantarflexionOverStiffness.xlsx", index=False)
    
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
    plt.title("Activation vs Length")
    plt.xlabel("Length cm")
    plt.ylabel("Activation")

    # Add grid, legend, and adjust layout
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # Save and display the plot
    plt.savefig("Solutions/Activation_vs_Stiffness_Combined.png")
    plt.show()

def calculateAverageActivationKnee(dir_path):
    # Define directory path

    # Define muscle groups
    Fmuscles = ['semiten_r', 'bifemlh_r', 'sar_r']
    Emuscles = ['rect_fem_r', 'vas_lat_r', 'vas_int_r']
                    
    # Create two DataFrames for the output Excel files
    extention_data = pd.DataFrame(columns=["stiffness", "activation"])
    flexion_data = pd.DataFrame(columns=["stiffness", "activation"])

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
                    if all(muscle in ACT.columns for muscle in Fmuscles + Emuscles):
                        # Calculate the average activation for Pmuscles and Dmuscles
                        avg_Fmuscles = ACT[Fmuscles].mean().mean()
                        avg_Emuscles = ACT[Emuscles].mean().mean()
                        
                        # Add data to the respective DataFrames
                        flexion_data = pd.concat([flexion_data, 
                                                    pd.DataFrame({"stiffness": [stiff], "activation": [avg_Fmuscles]})])
                        extention_data = pd.concat([extention_datas, 
                                                        pd.DataFrame({"stiffness": [stiff], "activation": [avg_Emuscles]})])
                    else:
                        print(f"File {file_name} is missing required muscle columns.")
                except Exception as e:
                    print(f"Error reading file {file_name}: {e}")
    # Save the results to Excel files
    
    flexion_data = flexion_data.sort_values(by='stiffness', ascending=True)
    flexion_data.reset_index(drop=True, inplace=True)
    flexion_data.to_excel("Solutions/ActivationsExls/KneeFlexionOverStiffness.xlsx", index=False)

    extention_data = extention_data.sort_values(by='stiffness', ascending=True)
    extention_data.reset_index(drop=True, inplace=True)
    extention_data.to_excel("Solutions/ActivationsExls/KneeExtensionOverStiffness.xlsx", index=False)
    
    merged_data = pd.merge(
        flexion_data,
        extention_data,
        on='stiffness',
        suffixes=('_flex', '_ext')
    )

    # Calculate the sum of activations
    merged_data['activation_sum'] = (
        merged_data['activation_flex'] + merged_data['activation_ext']
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
        flexion_data['stiffness'],
        flexion_data['activation'],
        label="Flextion",
        color='blue',
        marker='o'
    )

    # Plot Plantarflexion data
    plt.plot(
        extention_data['stiffness'],
        extention_data['activation'],
        label="Extension",
        color='green',
        marker='x'
    )

    # Add title and labels
    plt.title("Activation vs Stiffness For the Knee")
    plt.xlabel("Stiffness")
    plt.ylabel("Activation")

    # Add grid, legend, and adjust layout
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # Save and display the plot
    plt.savefig("Activation_vs_Stiffness_Combined_Knee.png")
    plt.show()

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

def plotSpringForce():
    file_10000 =  "Solutions/So/SO_10000_StaticOptimization_force.sto"
    file_90000 =  "Solutions/So/SO_90000_StaticOptimization_force.sto"
    file_61000 =  "Solutions/So/SO_61000_StaticOptimization_force.sto"
    
    ACT_10000 = pd.read_csv(file_10000, delim_whitespace=True, skiprows=14)  # Adjust the header skip count if necessary
    ACT_90000 = pd.read_csv(file_90000, delim_whitespace=True, skiprows=14)  # Adjust the header skip count if necessary
    ACT_61000 = pd.read_csv(file_61000, delim_whitespace=True, skiprows=14)  # Adjust the header skip count if necessary

    
    BiarticularSpringDamper_tension_10000 = ACT_10000["BiarticularSpringDamper_tension"]
    BiarticularSpringDamper_tension_90000 = ACT_90000["BiarticularSpringDamper_tension"]
    BiarticularSpringDamper_tension_61000 = ACT_61000["BiarticularSpringDamper_tension"]

    indices = np.arange(len(BiarticularSpringDamper_tension_10000))
    gaitCycle = indices / len(BiarticularSpringDamper_tension_10000) * 100

    plt.figure(figsize=(10, 8))
    plt.plot(gaitCycle,BiarticularSpringDamper_tension_10000, color="r", label = "K=10000")
    plt.plot(gaitCycle,BiarticularSpringDamper_tension_90000, color="b", label = "K=90000")
    plt.plot(gaitCycle,BiarticularSpringDamper_tension_61000, color="g", label = "K=61000")
    plt.title("Plot Spring force for different stiffness")
    plt.xlabel("Gayte cycle %")
    plt.ylabel("Force N")
    plt.tight_layout()
    plt.legend()

    plt.savefig("Solutions/plots/SpringForce/SpringForceVsStiffness.png", format='png', dpi=300)

def plotMainMuscleActivationDorsi():
    file_10000 =  "Solutions/So/SO_10000_StaticOptimization_activation.sto"
    file_90000 =  "Solutions/So/SO_90000_StaticOptimization_activation.sto"
    file_61000 =  "Solutions/So/SO_61000_StaticOptimization_activation.sto"
    
    ACT_10000 = pd.read_csv(file_10000, delim_whitespace=True, skiprows=8)  # Adjust the header skip count if necessary
    ACT_90000 = pd.read_csv(file_90000, delim_whitespace=True, skiprows=8)  # Adjust the header skip count if necessary
    ACT_61000 = pd.read_csv(file_61000, delim_whitespace=True, skiprows=8)  # Adjust the header skip count if necessary

    
    tib_ant_r_10000 = ACT_10000["tib_ant_r"]
    ext_dig_r_10000 = ACT_10000["ext_dig_r"]

    tib_ant_r_90000 = ACT_90000["tib_ant_r"]
    ext_dig_r_90000 = ACT_90000["ext_dig_r"]
    
    tib_ant_r_61000 = ACT_61000["tib_ant_r"]
    ext_dig_r_61000 = ACT_61000["ext_dig_r"]


    indices = np.arange(len(tib_ant_r_10000))
    gaitCycle = indices / len(tib_ant_r_10000) * 100

    plt.figure(figsize=(10, 8))
    plt.plot(gaitCycle,tib_ant_r_10000, color="r", label = "K=10000")
    plt.plot(gaitCycle,ext_dig_r_10000, color="r", label = "K=10000")
    
    plt.plot(gaitCycle,tib_ant_r_61000, color="g", label = "K=61000")
    plt.plot(gaitCycle,ext_dig_r_61000, color="g", label = "K=61000")

    plt.plot(gaitCycle,tib_ant_r_90000, color="b", label = "K=90000")
    plt.plot(gaitCycle,ext_dig_r_90000, color="b", label = "K=90000")
    
    plt.title("Activation on main Dorsiflexors with different stiffness")
    plt.xlabel("Gayte cycle %")
    plt.ylabel("Activation")
    plt.tight_layout()
    plt.legend()

    plt.savefig("Solutions/plots/ComparingPlots/ActivationVsStiffnessDorsi.png", format='png', dpi=300)


def plotMainMuscleActivationPlantar():
    file_10000 =  "Solutions/So/SO_10000_StaticOptimization_activation.sto"
    file_90000 =  "Solutions/So/SO_90000_StaticOptimization_activation.sto"
    file_61000 =  "Solutions/So/SO_61000_StaticOptimization_activation.sto"
    
    ACT_10000 = pd.read_csv(file_10000, delim_whitespace=True, skiprows=8)  # Adjust the header skip count if necessary
    ACT_90000 = pd.read_csv(file_90000, delim_whitespace=True, skiprows=8)  # Adjust the header skip count if necessary
    ACT_61000 = pd.read_csv(file_61000, delim_whitespace=True, skiprows=8)  # Adjust the header skip count if necessary

    
    med_gas_r_10000 = ACT_10000["med_gas_r"]
    soleus_r_10000 = ACT_10000["soleus_r"]

    med_gas_r_90000 = ACT_90000["med_gas_r"]
    soleus_r_90000 = ACT_90000["soleus_r"]
    
    med_gas_r_61000 = ACT_61000["med_gas_r"]
    soleus_r_61000 = ACT_61000["soleus_r"]


    indices = np.arange(len(med_gas_r_10000))
    gaitCycle = indices / len(med_gas_r_10000) * 100

    plt.figure(figsize=(10, 8))
    plt.plot(gaitCycle,med_gas_r_10000, color="r", label = "med_gasK=10000")
    plt.plot(gaitCycle,soleus_r_10000, color="r", label = "soleusK=10000")
    
    plt.plot(gaitCycle,med_gas_r_61000, color="g", label = "med_gasK=61000")
    plt.plot(gaitCycle,soleus_r_61000, color="g", label = "soleusK=61000")

    plt.plot(gaitCycle,med_gas_r_90000, color="b", label = "med_gasK=90000")
    plt.plot(gaitCycle,soleus_r_90000, color="b", label = "soleusK=90000")
    
    plt.title("Activation on main PlantarFlexors with different stiffness")
    plt.xlabel("Gayte cycle %")
    plt.ylabel("Activation")
    plt.tight_layout()
    plt.legend()

    plt.savefig("Solutions/plots/ComparingPlots/ActivationVsStiffnessPlantar.png", format='png', dpi=300)


def plotMainMuscleActivationKneeExt():
    file_10000 =  "Solutions/So/SO_10000_StaticOptimization_activation.sto"
    file_90000 =  "Solutions/So/SO_90000_StaticOptimization_activation.sto"
    file_61000 =  "Solutions/So/SO_61000_StaticOptimization_activation.sto"
    
    ACT_10000 = pd.read_csv(file_10000, delim_whitespace=True, skiprows=8)  # Adjust the header skip count if necessary
    ACT_90000 = pd.read_csv(file_90000, delim_whitespace=True, skiprows=8)  # Adjust the header skip count if necessary
    ACT_61000 = pd.read_csv(file_61000, delim_whitespace=True, skiprows=8)  # Adjust the header skip count if necessary

    
    rect_fem_r_10000 = ACT_10000["rect_fem_r"]
    vas_lat_r_10000 = ACT_10000["vas_lat_r"]
    vas_int_r_10000 = ACT_10000["vas_int_r"]


    rect_fem_r_90000 = ACT_90000["rect_fem_r"]
    vas_lat_r_90000 = ACT_90000["vas_lat_r"]
    vas_int_r_90000 = ACT_10000["vas_int_r"]


    rect_fem_r_61000 = ACT_61000["rect_fem_r"]
    vas_lat_r_61000 = ACT_61000["vas_lat_r"]
    vas_int_r_61000 = ACT_10000["vas_int_r"]


    indices = np.arange(len(rect_fem_r_61000))
    gaitCycle = indices / len(rect_fem_r_61000) * 100

    plt.figure(figsize=(10, 8))
    plt.plot(gaitCycle,rect_fem_r_10000, color="r", label = "rect_femK=10000")
    plt.plot(gaitCycle,vas_lat_r_10000, color="r", label = "vas_lat_rK=10000")
    plt.plot(gaitCycle,vas_int_r_10000, color="r", label = "vas_int_rK=10000")


    plt.plot(gaitCycle,rect_fem_r_61000, color="g", label = "rect_femK=61000")
    plt.plot(gaitCycle,vas_lat_r_61000, color="g", label = "vas_lat_rK=61000")
    plt.plot(gaitCycle,vas_int_r_61000, color="g", label = "vas_int_rK=61000")

    plt.plot(gaitCycle,rect_fem_r_90000, color="b", label = "rect_femK=90000")
    plt.plot(gaitCycle,vas_lat_r_90000, color="b", label = "vas_lat_rK=90000")
    plt.plot(gaitCycle,vas_int_r_90000, color="b", label = "vas_int_rK=90000")
    
    plt.title("Activation on main Knee Extensors with different stiffness")
    plt.xlabel("Gayte cycle %")
    plt.ylabel("Activation")
    plt.tight_layout()
    plt.legend()

    plt.savefig("Solutions/plots/ComparingPlots/ActivationVsStiffnessKneeExt.png", format='png', dpi=300)


def plotMainMuscleActivationKneeFlex():
    file_10000 =  "Solutions/So/SO_10000_StaticOptimization_activation.sto"
    file_90000 =  "Solutions/So/SO_90000_StaticOptimization_activation.sto"
    file_61000 =  "Solutions/So/SO_61000_StaticOptimization_activation.sto"
    
    ACT_10000 = pd.read_csv(file_10000, delim_whitespace=True, skiprows=8)  # Adjust the header skip count if necessary
    ACT_90000 = pd.read_csv(file_90000, delim_whitespace=True, skiprows=8)  # Adjust the header skip count if necessary
    ACT_61000 = pd.read_csv(file_61000, delim_whitespace=True, skiprows=8)  # Adjust the header skip count if necessary

    
    semiten_r_10000 = ACT_10000["semiten_r"]
    bifemlh_r_10000 = ACT_10000["bifemlh_r"]
    sar_r_10000 = ACT_10000["sar_r"]

    semiten_r_90000 = ACT_90000["semiten_r"]
    bifemlh_r_90000 = ACT_90000["bifemlh_r"]
    sar_r_90000 = ACT_10000["sar_r"]

    semiten_r_61000 = ACT_61000["semiten_r"]
    bifemlh_r_61000 = ACT_61000["bifemlh_r"]
    sar_r_61000 = ACT_10000["sar_r"]


    indices = np.arange(len(semiten_r_10000))
    gaitCycle = indices / len(semiten_r_10000) * 100

    plt.figure(figsize=(10, 8))
    plt.plot(gaitCycle,semiten_r_10000, color="r", label = "semitK=10000")
    plt.plot(gaitCycle,bifemlh_r_10000, color="r", label = "bifemK=10000")
    plt.plot(gaitCycle,sar_r_10000, color="r", label = "sar_rK=10000")

    plt.plot(gaitCycle,semiten_r_61000, color="g", label = "semitK=61000")
    plt.plot(gaitCycle,bifemlh_r_61000, color="g", label = "bifemK=61000")
    plt.plot(gaitCycle,sar_r_61000, color="g", label = "sar_rK=90000")

    plt.plot(gaitCycle,semiten_r_90000, color="b", label = "semitK=90000")
    plt.plot(gaitCycle,bifemlh_r_90000, color="b", label = "bifemK=90000")
    plt.plot(gaitCycle,sar_r_90000, color="b", label = "sar_rK=90000")

    plt.title("Activation on main Knee flexors with different stiffness")
    plt.xlabel("Gayte cycle %")
    plt.ylabel("Activation")
    plt.tight_layout()
    plt.legend()

    plt.savefig("Solutions/plots/ComparingPlots/ActivationVsStiffnessKneeFlex.png", format='png', dpi=300)

def plotAllDorsiFlection(isLen):
    output_path = ""

    if(isLen):
        output_path = "Solutions/plots/TotalActivationsLen/"
        path = "Solutions/SO/Len"
        value = 3
        colors = create_color_gradient(10)


    else:
        output_path = "Solutions/plots/TotalActivations/"
        path = "Solutions/SO"
        value = 1
        colors = create_color_gradient(70)


    labels = []  # To store the legend labels
    handles = []  # To store the corresponding handles
    # Plot the mean activation
    plt.figure(figsize=(10, 8))
    # Loop over each file in the directory
    counter = 0
    

    for idx,file_name in enumerate(os.listdir(path)):
        # Only process .sto files
        if file_name.endswith("activation.sto"):
            # Extract stiffness from the filename (assumed to be between underscores)
            stiffness = file_name.split('_')[value]
            print(file_name)
            # Construct full file path
            file_path = os.path.join(path, file_name)
            
            # Read the data from the file
            ACT = pd.read_csv(file_path, delim_whitespace=True, skiprows=8)
            
            # Extract muscle activation data
            tib_ant_r = ACT['tib_ant_r']
            ext_dig_r = ACT['ext_dig_r']
            ext_hal_r = ACT['ext_hal_r']
            per_tert_r = ACT['per_tert_r']
            mean = []
            for i in range(len(tib_ant_r)):
                mean.append((tib_ant_r[i] + ext_dig_r[i] + ext_hal_r[i] + per_tert_r[i])/4)
            
            # Create the gait cycle percentages
            indices = np.arange(len(mean))
            gait_cycle = indices / len(mean) * 100
            
            if(isLen):
                line, = plt.plot(gait_cycle, mean, color=colors[counter], label="0."+stiffness[:3])
                labels.append("0."+stiffness[:3])

            else:
                line, = plt.plot(gait_cycle, mean, color=colors[counter], label=stiffness)            
                labels.append(f"{stiffness}")

            handles.append(line)  # Save the line handle for legend
            counter +=1
    
    
    handles, labels = zip(*sorted(zip(handles, labels), key=lambda x: x[1]))
    if(isLen):
        plt.title(f"Mean Activation for Main Dorsiflexors over Length")
    else:
        plt.title(f"Mean Activation for Main Dorsiflexors over Stiffness")

    plt.xlabel("Gait Cycle (%)")
    plt.ylabel("Activation")
    plt.tight_layout()
    if(isLen):
        plt.legend(handles=handles, labels=labels, fontsize=10, ncol=1, bbox_to_anchor=(0.7, 0.75))
    else:
        plt.legend(handles=handles, labels=labels, fontsize=8, ncol=5, bbox_to_anchor=(0.3, 0.6))
            
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    if(isLen):            
        plt.savefig(f"{output_path}/TotalDorsiflexorsLen.png", format='png', dpi=300)
    else:
        plt.savefig(f"{output_path}/TotalDorsiflexorsStiffness.png", format='png', dpi=300)

    plt.close()  # Close the plot to free up memory


def plotAllPlantFlection(isLen):
    output_path = "Solutions/plots/TotalActivations"
    
    if(isLen):
        output_path = "Solutions/plots/TotalActivationsLen/"
        path = "Solutions/SO/Len"
        value = 3
        colors = create_color_gradient(10)


    else:
        output_path = "Solutions/plots/TotalActivations/"
        path = "Solutions/SO"
        value = 1
        colors = create_color_gradient(70)


    labels = []  # To store the legend labels
    handles = []  # To store the corresponding handles
    # Plot the mean activation
    plt.figure(figsize=(10, 8))
    # Loop over each file in the directory
    counter=0
    for idx,file_name in enumerate(os.listdir(path)):
        # Only process .sto files
        if file_name.endswith("activation.sto"):
            # Extract stiffness from the filename (assumed to be between underscores)
            stiffness = file_name.split('_')[value]
            
            # Construct full file path
            file_path = os.path.join(path, file_name)
            
            # Read the data from the file
            ACT = pd.read_csv(file_path, delim_whitespace=True, skiprows=8)
            
            # Extract muscle activation data
            med_gas_r = ACT['med_gas_r']
            soleus_r = ACT['soleus_r']
            tib_post_r = ACT['tib_post_r']
            per_brev_r = ACT['per_brev_r']
            per_long_r = ACT['per_long_r']
            mean = []
            for i in range(len(med_gas_r)):
                mean.append((med_gas_r[i] + soleus_r[i] + tib_post_r[i] + per_brev_r[i]+ per_long_r[i])/5)
            
            # Create the gait cycle percentages
            indices = np.arange(len(mean))
            gait_cycle = indices / len(mean) * 100
            
            if(isLen):
                line, = plt.plot(gait_cycle, mean, color=colors[counter], label="0."+stiffness[:3])
                labels.append("0."+stiffness[:3])

            else:
                line, = plt.plot(gait_cycle, mean, color=colors[counter], label=stiffness)            
                labels.append(f"{stiffness}")

            handles.append(line)  # Save the line handle for legend
            counter +=1
    
    
    handles, labels = zip(*sorted(zip(handles, labels), key=lambda x: x[1]))
    if(isLen):
        plt.title(f"Mean Activation for Main PlantarFlexors over Length")
    else:
        plt.title(f"Mean Activation for Main PlantarFlexors over Stiffness")

    plt.xlabel("Gait Cycle (%)")
    plt.ylabel("Activation")
    plt.tight_layout()
    if(isLen):
        plt.legend(handles=handles, labels=labels, fontsize=10, ncol=1, bbox_to_anchor=(0.7, 0.75))
    else:
        plt.legend(handles=handles, labels=labels, fontsize=8, ncol=3, bbox_to_anchor=(0.4, 0.35))
            
    if not os.path.exists(output_path):
        os.makedirs(output_path)            
    if(isLen): 
        plt.savefig(f"{output_path}/TotalPlantarflexorsLen.png", format='png', dpi=300)
    else:
        plt.savefig(f"{output_path}/TotalPlantarflexorsStiffness.png", format='png', dpi=300)

    plt.close()  # Close the plot to free up memory


def plotAllKneeFlection(isLen):
    output_path = "Solutions/plots/TotalActivations"
    
    if(isLen):
        output_path = "Solutions/plots/TotalActivationsLen/"
        path = "Solutions/SO/Len"
        value = 3
        colors = create_color_gradient(10)


    else:
        output_path = "Solutions/plots/TotalActivations/"
        path = "Solutions/SO"
        value = 1
        colors = create_color_gradient(70)

    labels = []  # To store the legend labels
    handles = []  # To store the corresponding handles
    # Plot the mean activation
    plt.figure(figsize=(10, 8))
    # Loop over each file in the directory
    counter=0
    for idx,file_name in enumerate(os.listdir(path)):
        # Only process .sto files
        if file_name.endswith("activation.sto"):
            # Extract stiffness from the filename (assumed to be between underscores)
            stiffness = file_name.split('_')[value]
            
            # Construct full file path
            file_path = os.path.join(path, file_name)
            
            # Read the data from the file
            ACT = pd.read_csv(file_path, delim_whitespace=True, skiprows=8)
            
            # Extract muscle activation data
            semiten_r = ACT['semiten_r']
            bifemlh_r = ACT['bifemlh_r']
            sar_r = ACT['sar_r']
            grac_r = ACT['grac_r']
            mean = []
            for i in range(len(semiten_r)):
                mean.append((semiten_r[i] + bifemlh_r[i] + sar_r[i] + grac_r[i])/4)
            
            # Create the gait cycle percentages
            indices = np.arange(len(mean))
            gait_cycle = indices / len(mean) * 100
            
            if(isLen):
                line, = plt.plot(gait_cycle, mean, color=colors[counter], label="0."+stiffness[:3])
                labels.append("0."+stiffness[:3])

            else:
                line, = plt.plot(gait_cycle, mean, color=colors[counter], label=stiffness)            
                labels.append(f"{stiffness}")

            handles.append(line)  # Save the line handle for legend
            counter +=1
    
    
    handles, labels = zip(*sorted(zip(handles, labels), key=lambda x: x[1]))
    if(isLen):
        plt.title(f"Mean Activation for Main Knee Flexors over Length")
    else:
        plt.title(f"Mean Activation for Main Knee Flexors over Stiffness")

    plt.xlabel("Gait Cycle (%)")
    plt.ylabel("Activation")
    plt.tight_layout()
    
    if(isLen):
        plt.legend(handles=handles, labels=labels, fontsize=10, ncol=1, bbox_to_anchor=(0.7, 0.75))
    else:
        plt.legend(handles=handles, labels=labels, fontsize=8, ncol=5, bbox_to_anchor=(0.3, 0.6))
            
    if not os.path.exists(output_path):
        os.makedirs(output_path)    
    if(isLen):        
        plt.savefig(f"{output_path}/TotalKneeflexorsLen.png", format='png', dpi=300)
    else:
        plt.savefig(f"{output_path}/TotalKneeflexorsStiff.png", format='png', dpi=300)

    plt.close()  # Close the plot to free up memory


def plotAllKneeExtension(isLen):
    output_path = "Solutions/plots/TotalActivations"
    
    if(isLen):
        output_path = "Solutions/plots/TotalActivationsLen/"
        path = "Solutions/SO/Len"
        value = 3
        colors = create_color_gradient(10)


    else:
        output_path = "Solutions/plots/TotalActivations/"
        path = "Solutions/SO"
        value = 1
        colors = create_color_gradient(70)


    labels = []  # To store the legend labels
    handles = []  # To store the corresponding handles
    # Plot the mean activation
    plt.figure(figsize=(10, 8))
    # Loop over each file in the directory
    counter=0
    for idx,file_name in enumerate(os.listdir(path)):
        # Only process .sto files
        if file_name.endswith("activation.sto"):
            # Extract stiffness from the filename (assumed to be between underscores)
            stiffness = file_name.split('_')[value]
            
            # Construct full file path
            file_path = os.path.join(path, file_name)
            
            # Read the data from the file
            ACT = pd.read_csv(file_path, delim_whitespace=True, skiprows=8)
            
            # Extract muscle activation data
            rect_fem_r = ACT['rect_fem_r']
            vas_lat_r = ACT['vas_lat_r']
            vas_int_r = ACT['vas_int_r']
            mean = []
            for i in range(len(rect_fem_r)):
                mean.append((rect_fem_r[i] + vas_lat_r[i] + vas_int_r[i])/3)
            
            # Create the gait cycle percentages
            indices = np.arange(len(mean))
            gait_cycle = indices / len(mean) * 100
            
            if(isLen):
                line, = plt.plot(gait_cycle, mean, color=colors[counter], label="0."+stiffness[:3])
                labels.append("0."+stiffness[:3])

            else:
                line, = plt.plot(gait_cycle, mean, color=colors[counter], label=stiffness)            
                labels.append(f"{stiffness}")

            handles.append(line)  # Save the line handle for legend
            counter +=1
    
    
    handles, labels = zip(*sorted(zip(handles, labels), key=lambda x: x[1]))
    if(isLen):
        plt.title(f"Mean Activation for Main Knee Extensors over Length")
    else:
        plt.title(f"Mean Activation for Main Knee Extensors over Stiffness")

    plt.xlabel("Gait Cycle (%)")
    plt.ylabel("Activation")
    plt.tight_layout()
    
    if(isLen):
        plt.legend(handles=handles, labels=labels, fontsize=10, ncol=1, bbox_to_anchor=(0.7, 0.75))
    else:
        plt.legend(handles=handles, labels=labels, fontsize=8, ncol=3, bbox_to_anchor=(0.7, 0.4))
            
    if not os.path.exists(output_path):
        os.makedirs(output_path)      
    if(isLen):      
        plt.savefig(f"{output_path}/TotalKneeExtensorlen.png", format='png', dpi=300)
    else:
        plt.savefig(f"{output_path}/TotalKneeExtensorStiffness.png", format='png', dpi=300)

    plt.close()  # Close the plot to free up memory


def plotAllHipExtension(isLen):
    output_path = "Solutions/plots/TotalActivations"
    
    if(isLen):
        output_path = "Solutions/plots/TotalActivationsLen/"
        path = "Solutions/SO/Len"
        value = 3
        colors = create_color_gradient(10)


    else:
        output_path = "Solutions/plots/TotalActivations/"
        path = "Solutions/SO"
        value = 1
        colors = create_color_gradient(70)


    labels = []  # To store the legend labels
    handles = []  # To store the corresponding handles
    # Plot the mean activation
    plt.figure(figsize=(10, 8))
    # Loop over each file in the directory
    counter=0
    for idx,file_name in enumerate(os.listdir(path)):
        # Only process .sto files
        if file_name.endswith("activation.sto"):
            # Extract stiffness from the filename (assumed to be between underscores)
            stiffness = file_name.split('_')[value]
            
            # Construct full file path
            file_path = os.path.join(path, file_name)
            
            # Read the data from the file
            ACT = pd.read_csv(file_path, delim_whitespace=True, skiprows=8)
            
            # Extract muscle activation data
            glut_max1_r = ACT['glut_max1_r']
            glut_max2_r = ACT['glut_max2_r']
            glut_max3_r = ACT['glut_max3_r']
            bifemsh_r = ACT['bifemsh_r']
            semimem_r = ACT['semimem_r']
            grac_r = ACT['grac_r']
            mean = []
            for i in range(len(glut_max1_r)):
                mean.append((glut_max1_r[i] + glut_max2_r[i] + glut_max3_r[i]+ bifemsh_r[i]+semimem_r[i] + grac_r[i])/6)
            
            # Create the gait cycle percentages
            indices = np.arange(len(mean))
            gait_cycle = indices / len(mean) * 100
            
            if(isLen):
                line, = plt.plot(gait_cycle, mean, color=colors[counter], label="0."+stiffness[:3])
                labels.append("K=0."+stiffness[:3])

            else:
                line, = plt.plot(gait_cycle, mean, color=colors[counter], label=stiffness)            
                labels.append(f"K={stiffness}")

            handles.append(line)  # Save the line handle for legend
            counter +=1
    
    
    handles, labels = zip(*sorted(zip(handles, labels), key=lambda x: x[1]))
    if(isLen):
        plt.title(f"Mean Activation for Main Hip Extensors over Length")
    else:
        plt.title(f"Mean Activation for Main Hip Extensors over Stiffness")

    plt.xlabel("Gait Cycle (%)")
    plt.ylabel("Activation")
    plt.tight_layout()
    
    if(isLen):
        plt.legend(handles=handles, labels=labels, fontsize=10, ncol=1, bbox_to_anchor=(0.7, 0.75))
    else:
        plt.legend(handles=handles, labels=labels, fontsize=8, ncol=5, bbox_to_anchor=(0.3, 0.6))
            
    if not os.path.exists(output_path):
        os.makedirs(output_path)        
    if(isLen):    
        plt.savefig(f"{output_path}/TotalHipExtensorStiffness.png", format='png', dpi=300)
    else:
        plt.savefig(f"{output_path}/TotalHipExtensorLen.png", format='png', dpi=300)

    plt.close()  # Close the plot to free up memory


def plotAllHipFlexion(isLen):
    output_path = "Solutions/plots/TotalActivations"
    
    if(isLen):
        output_path = "Solutions/plots/TotalActivationsLen/"
        path = "Solutions/SO/Len"
        value = 3
        colors = create_color_gradient(10)


    else:
        output_path = "Solutions/plots/TotalActivations/"
        path = "Solutions/SO"
        value = 1
        colors = create_color_gradient(70)


    labels = []  # To store the legend labels
    handles = []  # To store the corresponding handles
    # Plot the mean activation
    plt.figure(figsize=(10, 8))
    # Loop over each file in the directory
    counter=0
    for idx,file_name in enumerate(os.listdir(path)):
        # Only process .sto files
        if file_name.endswith("activation.sto"):
            # Extract stiffness from the filename (assumed to be between underscores)
            stiffness = file_name.split('_')[value]
            
            # Construct full file path
            file_path = os.path.join(path, file_name)
            
            # Read the data from the file
            ACT = pd.read_csv(file_path, delim_whitespace=True, skiprows=8)
            
            # Extract muscle activation data
            rect_fem_r = ACT['rect_fem_r']
            iliacus_r = ACT['iliacus_r']
            psoas_r = ACT['psoas_r']
            sar_r = ACT['sar_r']
            mean = []
            for i in range(len(rect_fem_r)):
                mean.append((rect_fem_r[i] + iliacus_r[i] + psoas_r[i]+ sar_r[i])/3)
            
            # Create the gait cycle percentages
            indices = np.arange(len(mean))
            gait_cycle = indices / len(mean) * 100

            if(isLen):
                line, = plt.plot(gait_cycle, mean, color=colors[counter], label="0."+stiffness[:3])
                labels.append("0."+stiffness[:3])

            else:
                line, = plt.plot(gait_cycle, mean, color=colors[counter], label=stiffness)            
                labels.append(f"{stiffness}")

            handles.append(line)  # Save the line handle for legend
            counter +=1
    
    
    handles, labels = zip(*sorted(zip(handles, labels), key=lambda x: x[1]))
    if(isLen):
        plt.title(f"Mean Activation for Main Hip Flexors over Legth")
    else:
        plt.title(f"Mean Activation for Main Hip Flexors over Stiffness")

    plt.xlabel("Gait Cycle (%)")
    plt.ylabel("Activation")
    plt.tight_layout()
    
    if(isLen):
        plt.legend(handles=handles, labels=labels, fontsize=10, ncol=1, bbox_to_anchor=(0.7, 0.75))
    else:
        plt.legend(handles=handles, labels=labels, fontsize=10, ncol=3, bbox_to_anchor=(0.4, 0.35))
            
    if not os.path.exists(output_path):
        os.makedirs(output_path)            
    if(isLen):
        plt.savefig(f"{output_path}/TotalHipFlexorLength.png", format='png', dpi=300)
    else:
        plt.savefig(f"{output_path}/TotalHipFlexorStiffness.png", format='png', dpi=300)

    plt.close()  # Close the plot to free up memory


def plotAllPowerSpring(isLen):
    output_path = "Solutions/plots/TotalActivations"
    
    if(isLen):
        output_path = "Solutions/plots/TotalActivationsLen/"
        path = "Solutions/SO/Len"
        value = 3
        colors = create_color_gradient(10)


    else:
        output_path = "Solutions/plots/TotalActivations/"
        path = "Solutions/SO"
        value = 1
        colors = create_color_gradient(70)


    labels = []  # To store the legend labels
    handles = []  # To store the corresponding handles
    # Plot the mean activation
    plt.figure(figsize=(10, 8))
    # Loop over each file in the directory
    counter=0
    for idx,file_name in enumerate(os.listdir(path)):
        # Only process .sto files
        if file_name.endswith("force.sto"):
            # Extract stiffness from the filename (assumed to be between underscores)
            stiffness = file_name.split('_')[value]
            
            # Construct full file path
            file_path = os.path.join(path, file_name)
            
            # Read the data from the file
            ACT = pd.read_csv(file_path, delim_whitespace=True, skiprows=14)
            
            # Extract muscle activation data
            BiarticularSpringDamper_tension = ACT['BiarticularSpringDamper_tension']
           
        
            # Create the gait cycle percentages
            indices = np.arange(len(BiarticularSpringDamper_tension))
            gait_cycle = indices / len(BiarticularSpringDamper_tension) * 100
            
            if(isLen):
                line, = plt.plot(gait_cycle, BiarticularSpringDamper_tension, color=colors[counter], label="0."+stiffness[:3])
                labels.append("0."+stiffness[:3])

            else:
                line, = plt.plot(gait_cycle, BiarticularSpringDamper_tension, color=colors[counter], label=stiffness)            
                labels.append(f"{stiffness}")

            handles.append(line)  # Save the line handle for legend
            counter +=1
    
    
    handles, labels = zip(*sorted(zip(handles, labels), key=lambda x: x[1]))
    if(isLen):
        plt.title(f"Spring force over Length")
    else:
        plt.title(f"Spring force over Stiffness")

    plt.xlabel("Gait Cycle (%)")
    plt.ylabel("N")
    plt.tight_layout()
    if(isLen):
        plt.legend(handles=handles, labels=labels, fontsize=10, ncol=1, bbox_to_anchor=(0.85, 0.55))
    else:
        plt.legend(handles=handles, labels=labels, fontsize=8, ncol=3, bbox_to_anchor=(0.6, 0.45))
            
    if not os.path.exists(output_path):
        os.makedirs(output_path)            
    if(isLen): 
        plt.savefig(f"{output_path}/TotalSpringPowerLen.png", format='png', dpi=300)
    else:
        plt.savefig(f"{output_path}/TotalSpringPowerStiff.png", format='png', dpi=300)

    plt.close()  # Close the plot to free up memory


def plotAllActivationForFile(file_path,output_path,stiffness):
    # Load the data
    ACT = pd.read_csv(file_path, delim_whitespace=True, skiprows=8)
    colors = create_color_gradient(2)
    # Function to calculate the mean activation for a set of columns
    def calculate_mean(values):
        return np.mean(values, axis=0)
    def calculate_sum(values):
        return np.sum(values, axis=0)
    # Create gait cycle
    indices = np.arange(len(ACT['semiten_r']))  # Assuming all columns have the same length
    gaitCycle = indices / len(indices) * 100

    # Calculate mean activations for each group
    # Dorsiflexion
    dorsiflexion_values = np.vstack([ACT['tib_ant_r'], ACT['ext_dig_r'], ACT['ext_hal_r'], ACT['per_tert_r']])
    mean_dorsiflexion = calculate_mean(dorsiflexion_values)

    # Plantarflexion
    plantarflexion_values = np.vstack([ACT['med_gas_r'], ACT['soleus_r'], ACT['tib_post_r'], ACT['per_brev_r'], ACT['per_long_r']])
    mean_plantarflexion = calculate_mean(plantarflexion_values)

    # Knee Flexion
    knee_flexion_values = np.vstack([ACT['semiten_r'], ACT['bifemlh_r'], ACT['sar_r'], ACT['grac_r']])
    mean_knee_flexion = calculate_mean(knee_flexion_values)

    # Knee Extension
    knee_extension_values = np.vstack([ACT['rect_fem_r'], ACT['vas_lat_r'], ACT['vas_int_r']])
    mean_knee_extension = calculate_mean(knee_extension_values)

    # Hip Flexion
    hip_flexion_values = np.vstack([ACT['rect_fem_r'], ACT['iliacus_r'], ACT['psoas_r'], ACT['sar_r']])
    mean_hip_flexion = calculate_mean(hip_flexion_values)

    # Hip Extension
    hip_extension_values = np.vstack([ACT['glut_max1_r'], ACT['glut_max2_r'], ACT['glut_max3_r'], 
                                    ACT['bifemsh_r'], ACT['semimem_r'], ACT['grac_r']])
    mean_hip_extension = calculate_mean(hip_extension_values)

    # Plot all mean activations on the same graph
    plt.figure(figsize=(10, 6))

    plt.plot(gaitCycle, mean_dorsiflexion,color=colors[0], label='Dorsiflexion')
    plt.plot(gaitCycle, mean_plantarflexion,color=colors[1], label='Plantarflexion')
    # plt.plot(gaitCycle, mean_knee_flexion,color=colors[0], label='Knee Flexion')
    # plt.plot(gaitCycle, mean_knee_extension,color=colors[1], label='Knee Extension')
    # plt.plot(gaitCycle, mean_hip_flexion,color=colors[0], label='Hip Flexion')
    # plt.plot(gaitCycle, mean_hip_extension,color=colors[1], label='Hip Extension')

    # Customize the plot
    plt.title('Average Activation with 3cm change in geometry')
    plt.xlabel('Gait Cycle (%)')
    plt.ylabel('Mean Activation')
    plt.legend(fontsize=10, ncol=1, bbox_to_anchor=(0.8, 0.65))
    plt.grid(True)

    # Show the plot
    plt.savefig(output_path+"/FootGeostiff"+stiffness+"Stiffness.png", format='png', dpi=300)


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
    plt.title("Plot Main DorsiFlexors For 3cm attachment")
    plt.xlabel("Gayte cycle %")
    plt.ylabel("Activation")
    plt.tight_layout()
    plt.legend()

    plt.savefig(output_path+"/DorsiflexorsGeo"+stiffness+"Stiffness.png", format='png', dpi=300)



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
    plt.title("Plot Main Plantarflexors For 3cm attachment")
    plt.xlabel("Gayte cycle %")
    plt.ylabel("Activation")
    plt.tight_layout()
    plt.legend()

    plt.savefig(output_path+"/PlantarflexorsGeo"+stiffness+"Stiffness.png", format='png', dpi=300)

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
    plt.title("Plot Main Knee Flexors For 3cm attachment")
    plt.xlabel("Gayte cycle %")
    plt.ylabel("Activation")
    plt.tight_layout()
    plt.legend()

    plt.savefig(output_path+"/KneeFlexionGeoFor"+stiffness+"Stiffness.png", format='png', dpi=300)

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
    plt.title("Plot Main Knee Extensors For 3cm attachment")
    plt.xlabel("Gayte cycle %")
    plt.ylabel("Activation")
    plt.tight_layout()
    plt.legend()

    plt.savefig(output_path+"/KneeExtensionGeoFor"+stiffness+"Stiffness.png", format='png', dpi=300)

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
    plt.title("Plot Main Hip Flexors For 3cm attachment")
    plt.xlabel("Gayte cycle %")
    plt.ylabel("Activation")
    plt.tight_layout()
    plt.legend()

    plt.savefig(output_path+"/HipFlexionGeoFor"+stiffness+"Stiffness.png", format='png', dpi=300)

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

    plt.title("Plot Main Hip Extensors For 3cm attachment")
    plt.xlabel("Gayte cycle %")
    plt.ylabel("Activation")
    # Set the background color and display the plots
    plt.tight_layout()
    plt.legend()

    plt.savefig(output_path+"/HipExtensionGeoFor"+stiffness+"Stiffness.png", format='png', dpi=300)


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

def PartA(startingValue,endingValue,step):
    i = startingValue
    while i < endingValue:
        # Set geometry path for the path spring to match the gastrocnemius muscle 
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


def PartA(startingValue,endingValue,step):
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


def PartB(startingValue,endingValue,step):
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

#PartA(int(startingValue),int(endingValue),float(step))
#PartB(float(startingValue),float(endingValue),float(step))

#plotAll("Solutions/SO/","0_32")
#plotAllInDir()
#calculateAverageActivation("Solutions/SO/Len")
#createPAndDFlexionXml("Solutions/SO/Len")
#plotSpringForce()
# plotMainMuscleActivationDorsi()
# plotMainMuscleActivationPlantar()
# plotMainMuscleActivationKneeFlex()
# plotMainMuscleActivationKneeExt()
#calculateAverageActivationKnee("Solutions/SO/Len")
# plotAllDorsiFlection(False)
# plotAllPlantFlection(False)
# plotAllKneeFlection(False)
# plotAllKneeExtension(False)
# plotAllHipExtension(True)
# plotAllHipFlexion(True)
#plotAllPowerSpring(True)

# plotPlantarFlection("Solutions/SO/Geo/Geo_SO_61000_327_StaticOptimization_activation.sto","Solutions/plots/Geo","61000")
# plotDorsiFlection("Solutions/SO/Geo/Geo_SO_61000_327_StaticOptimization_activation.sto","Solutions/plots/Geo","61000")
# plotKneeExtension("Solutions/SO/Geo/Geo_SO_61000_327_StaticOptimization_activation.sto","Solutions/plots/Geo","61000")
# plotKneeFlection("Solutions/SO/Geo/Geo_SO_61000_327_StaticOptimization_activation.sto","Solutions/plots/Geo","61000")
# plotHipExtension("Solutions/SO/Geo/Geo_SO_61000_327_StaticOptimization_activation.sto","Solutions/plots/Geo","61000")
# plotHipFlexion("Solutions/SO/Geo/Geo_SO_61000_327_StaticOptimization_activation.sto","Solutions/plots/Geo","61000")
plotAllActivationForFile("Solutions/SO/Geo/Geo_SO_61000_327_StaticOptimization_activation.sto","Solutions/plots/Geo","61000")