import yaml
import re

# Function to replace placeholders in the template file with values from the parameters file
def replace_placeholders(template, params):
    # Define a regex pattern for placeholders in the format <KEY>
    pattern = re.compile(r"<(.*?)>")
    
    def replacer(match):
        key = match.group(1)  # Extract the key inside the angle brackets
        return str(params.get(key, match.group(0)))  # Replace with value from params, or keep original if key not found
    
    return pattern.sub(replacer, template)

# File paths
template_file = "D03_hm_tmpl.yaml"
params_file = "D03_params.yaml"
output_file = "config_sim_D03.yaml"

# Read the template file
with open(template_file, 'r') as tmpl_file:
    template_content = tmpl_file.read()

# Read the parameters file
with open(params_file, 'r') as params_file:
    params = yaml.safe_load(params_file)

# Replace placeholders in the template
processed_content = replace_placeholders(template_content, params)

# Save the output to a new file
with open(output_file, 'w') as output:
    output.write(processed_content)

print(f"Processed content saved to {output_file}")
