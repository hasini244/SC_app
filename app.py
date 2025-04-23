import pandas as pd
import numpy as np
import re
import joblib
from flask import Flask, request, render_template, jsonify
from flask import Flask, render_template, request
import statistics
app = Flask(__name__)

lgbm_regressor = joblib.load('model.pkl')  # Load pre-trained model

# Read the atomic radius and other element-related data
df_atomic_radius = pd.read_csv('allfeatures final.csv', index_col='symbol')
df = pd.read_csv('data/allfeatures final.csv')  # Update with our correct CSV file path

# Function to extract elements and their values from a chemical formula
def extract_elements(formula):
    matches = re.findall(r'([A-Z][a-z]*)([0-9.+-]*)', formula)
    elements_dict = {element: float(value) if value else 1.0 for element, value in matches}
    return elements_dict

# Function to calculate weighted mean
def calculate_weighted_mean(element_values, variable):
    total_value = sum(element_values.values())
    x_values = {element: element_values[element] / total_value for element in element_values}
    weighted_mean = sum(x * df_atomic_radius.loc[element, variable] for element, x in x_values.items())
    return weighted_mean

# Function to calculate range
def calculate_range(element_values, variable):
    values = [df_atomic_radius.loc[element, variable] for element in element_values]
    variable_range = max(values) - min(values)
    return variable_range

# Function to calculate weighted standard deviation
def calculate_weighted_std_dev(element_values, variable):
    total_value = sum(element_values.values())
    x_values = {element: element_values[element] / total_value for element in element_values}
    weighted_mean = sum(x * df_atomic_radius.loc[element, variable] for element, x in x_values.items())
    weighted_std_dev = np.sqrt(sum(x * (df_atomic_radius.loc[element, variable] - weighted_mean)**2 for element, x in x_values.items()))
    return weighted_std_dev

# Function to calculate max
def calculate_max(element_values, variable):
    values = [df_atomic_radius.loc[element, variable] for element in element_values]
    variable_max = max(values)
    return variable_max

# Function to calculate min
def calculate_min(element_values, variable):
    values = [df_atomic_radius.loc[element, variable] for element in element_values]
    variable_min = min(values)
    return variable_min

# Function to calculate mode
def calculate_mode(element_values, variable):
    values = [df_atomic_radius.loc[element, variable] for element in element_values]
    return statistics.mode(values)


def extract_element_values(material):
    # Initialize a dictionary to store element values
    element_values = {}

    # Use regular expression to find element symbols and values
    pattern = re.compile(r'([A-Z][a-z]*)([+-]?\d*\.?\d+)?')
    matches = pattern.findall(material)

    # Iterate through matches and update the dictionary
    for match in matches:
        element, value = match
        value = float(value) if value else 1.0
        element_values[element] = value

    return element_values

# Read the thermal conductivity values from the thermal conductivity file
thermal_conductivity_df = pd.read_csv('allfeatures final.csv')

# Create a dictionary from the thermal conductivity data
thermal_conductivity_values = dict(zip(thermal_conductivity_df['symbol'],
                                       thermal_conductivity_df['thermal conductivity']))

# Maximum number of elements in material
max_elements = 9

# Extract features based on the chemical formula of the material
def extract_features(material_formula):
    # Extract element values from the material formula
    element_values = extract_elements(material_formula)

    features = {}  # Initialize features as a dictionary
    
    # Calculate the features dynamically based on extracted element values
    for variable in df_atomic_radius.columns:
        weighted_mean = calculate_weighted_mean(element_values, variable)
        range_val = calculate_range(element_values, variable)
        weighted_std_dev = calculate_weighted_std_dev(element_values, variable)
        variable_max = calculate_max(element_values, variable)
        variable_min = calculate_min(element_values, variable)
        variable_mode = calculate_mode(element_values, variable)
        
        # Create feature names and add them to the dictionary
        features[f'weighted_mean_{variable}'] = weighted_mean
        features[f'{variable}_range'] = range_val
        features[f'weighted_std_dev_{variable}'] = weighted_std_dev
        features[f'{variable}_max'] = variable_max
        features[f'{variable}_min'] = variable_min
        features[f'{variable}_mode'] = variable_mode

    return features


# Flask setup to create an API (if you're using Flask for web service)
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the material formula from the incoming request
    material = request.json['material']
    print("Received material:", material)  # Debugging print
    # Predict the critical temperature for the material
    features = extract_features(material)
    # print("Extracted features:", features)  # Debugging print
    # Extract element values from the material string
    element_values = extract_element_values(material)

    # Create a dictionary to store the input data
    row_data = {'material': material}
    dfs=[]
    # Populate the row_data dictionary with element values multiplied by thermal conductivity values
    for i in range(max_elements):
        element = 'C{}'.format(i + 1)
        if i < len(element_values):
            element_value = list(element_values.values())[i]
            element_symbol = list(element_values.keys())[i]
            thermal_conductivity_value = thermal_conductivity_values.get(element_symbol, 0)
            row_data[element] = element_value * thermal_conductivity_value
        else:
            row_data[element] = 0
    dfs.append(pd.DataFrame([row_data]))
    # Combine dfs into a single DataFrame (if multiple rows are expected)
    final_df = pd.concat(dfs, ignore_index=True)

    # Convert the first row to a dictionary
    final_dict = final_df.iloc[0].to_dict()

    # Log the result for debugging
    # print("Final dictionary:", final_dict)

    # Extract composition from the input material
    composition = extract_elements(material)
    material_df = df[df['symbol'].isin(composition.keys())]

    # Initialize values for specific elements of interest (e.g., 'Cu' and 'O')
    specific_elements = ['Cu', 'O']
    element_weights = {element: 0 for element in specific_elements}
    element_specific_heats = {element: 0 for element in specific_elements}

    # Calculate the element weights and specific heats
    for element in specific_elements:
        if element in composition:
            total_formula_weight = sum(material_df.loc[material_df['symbol'] == el, 'atomic weight'].values[0] * count for el, count in composition.items())
            total_formula_specific_heat = sum(material_df.loc[material_df['symbol'] == el, 'specific heat'].values[0] * count for el, count in composition.items())

            # Calculate the element weight
            element_weights[element] = 100 * (composition[element] * material_df.loc[material_df['symbol'] == element, 'atomic weight'].values[0]) / total_formula_weight

            # Calculate the element specific heat
            element_specific_heats[element] = 100 * (composition[element] * material_df.loc[material_df['symbol'] == element, 'specific heat'].values[0]) / total_formula_specific_heat

    # Prepare the results to send back to the user
    result = {
        'material': material,
        **{f'{element} Weight': weight for element, weight in element_weights.items()},
        **{f'{element} Specific Heat': specific_heat for element, specific_heat in element_specific_heats.items()}
    } 
    element_pattern = r'([A-Z][a-z]*)(\d*\.?\d*)'

    # Find all elements in the material string
    elements = re.findall(element_pattern, material)
    number_of_elemnts=len(elements)

    # Flatten dictionary: include keys and values
    flattened_features = [item for pair in features.items() for item in pair]
    final_df_dict = final_df.iloc[0].to_dict()

        # Combine everything into a single dictionary
    full_dataset_features = {
        **dict(zip(flattened_features[::2], flattened_features[1::2])),  # Convert flattened features back to a dictionary
        'number_of_elements': number_of_elemnts,  # Add number_of_elements as a feature
        **result,                                 # Add result values
        **final_df_dict                          # Add features from final_df with their names
    }

    # Print the combined dictionary
    # print("full_dataset_features",full_dataset_features)


# Assuming full_dataset_features_df is your DataFrame containing all features
    full_dataset_features_df = pd.DataFrame(full_dataset_features, index=[0])  # Add an index

    selected_features = [
    "thermal conductivity_range", "weighted_std_dev_space group number", 
    "weighted_std_dev_GSvolume_pa", "weighted_std_dev_GSmagmom", "GSmagmom_range", "C5", 
    "O Specific Heat", "weighted_mean_atomic weight", "weighted_mean_NUnfilled", 
    "weighted_std_dev_NfUnfilled", "density_min", "C3", "weighted_std_dev_NUnfilled", 
    "weighted_mean_melting point", "weighted_std_dev_thermal conductivity", 
    "weighted_mean_Electron affinity", "weighted_std_dev_NpUnfilled", 
    "weighted_std_dev_MendeleevNumber", "Cu Weight", "weighted_std_dev_cohensive energy", 
    "weighted_mean_boiling point", "weighted_std_dev_boiling point", 
    "weighted_mean_thermal conductivity", "weighted_std_dev_heat of fusion", "C4", 
    "weighted_std_dev_electronegativity", "weighted_std_dev_Electron affinity", 
    "weighted_std_dev_Ionization Energy", "weighted_std_dev_atomic radius", 
    "space group number_max", "weighted_std_dev_Column", "weighted_mean_density", 
    "weighted_mean_heat of vaporization", "Cu Specific Heat", "weighted_mean_electronegativity", 
    "weighted_mean_Ionization Energy", "weighted_mean_NValence", "C1", "weighted_std_dev_density", 
    "weighted_mean_MendeleevNumber", "weighted_mean_atomic radius", "weighted_mean_crystal radius", 
    "weighted_std_dev_NValence", "weighted_mean_cohensive energy", "weighted_std_dev_NdValence", 
    "weighted_mean_heat of fusion", "weighted_std_dev_heat of vaporization", "Electron affinity_min", 
    "weighted_std_dev_crystal radius", "C2", "weighted_mean_specific heat", "weighted_mean_GSvolume_pa", 
    "weighted_mean_Column", "weighted_mean_Number", "weighted_std_dev_specific heat", 
    "weighted_std_dev_melting point", "weighted_mean_space group number", "C6", 
    "weighted_std_dev_Ionic radius", "weighted_mean_Ionic radius", "weighted_std_dev_NdUnfilled", 
    "weighted_mean_NdValence", "weighted_mean_NdUnfilled", "O Weight", "weighted_std_dev_atomic weight", 
    "weighted_std_dev_Number", "weighted_std_dev_Row", "electronegativity_min", 
    "weighted_mean_Row", "Ionization Energy_mode"
]
# Select only the 71 features
    selected_df = full_dataset_features_df[selected_features]
    print("selected_df",selected_df)
# Now `selected_df` will contain only the selected 71 features
# Append material type (0, 1, or 2) for prediction
    material_type = 0  # Example: Type 0
    selected_df1 = selected_df.copy()
    selected_df1.insert(0, 'material_type', material_type)
    non_predicted_temp = lgbm_regressor.predict(selected_df1)
    non_predicted_temp = np.clip(non_predicted_temp, 0, None)
    print("non_predicted_temp",non_predicted_temp)

    material_type = 1  # Example: Type 1
    selected_df2 = selected_df.copy()
    selected_df2.insert(0, 'material_type', material_type)
    low_predicted_temp = lgbm_regressor.predict(selected_df2)
    print("low_predicted_temp",low_predicted_temp)

    material_type = 2  # Example: Type 2
    selected_df3 = selected_df.copy()
    selected_df3.insert(0, 'material_type', material_type)
    high_predicted_temp = lgbm_regressor.predict(selected_df3)
    print("high_predicted_temp",high_predicted_temp)


    return jsonify({
        "non_superconductor_temp":non_predicted_temp[0],
        "low_predicted_temp": low_predicted_temp[0],
        "high_predicted_temp": high_predicted_temp[0]
    })

if __name__ == '__main__':
    app.run(debug=True)
