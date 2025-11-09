"""A profile to run the CSC2235 project.

Instructions:
Wait for the node to be "Ready". This means the setup script
has finished installing pip and all Python requirements.
Your project code is in /local/repository.
"""

import geni.portal as portal
import geni.rspec.pg as rspec

# --- Define a user-selectable parameter for hardware type (Section 8.8) ---
portal.context.defineParameter(
    "hwtype", "Physical Node Type",
    portal.ParameterType.STRING, "c220g5",
    [('c220g5', 'Wisconsin: c220g5 (Intel Skylake)'),
     ('c240g5', 'Wisconsin: c240g5 (Intel Skylake)')],
    "Select an x86-based physical node type. If your first choice is unavailable, try another."
)

# Retrieve the values the user specifies during instantiation
params = portal.context.bindParameters()

# Create a Request object
request = portal.context.makeRequestRSpec()

# Add a single raw PC (physical machine)
node = request.RawPC("node")

# Specify the OS image (Ubuntu 22.04 is the default for this node)
node.disk_image = "urn:publicid:IDN+emulab.net+image+emulab-ops//UBUNTU24-64-STD"

# --- Use the parameter value ---
# Request the hardware type selected by the user from the dropdown
node.hardware_type = params.hwtype

# Add an "execute service" to run your setup script
node.addService(rspec.Execute(shell="bash", 
                              command="/local/repository/setup.sh"))

# Print the RSpec
portal.context.printRequestRSpec()