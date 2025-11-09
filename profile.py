"""A profile to run the CSC2235 project.

Instructions:
Wait for the node to be "Ready". This means the setup script
has finished installing pip and all Python requirements.
Your project code is in /local/repository.
"""

import geni.portal as portal
import geni.rspec.pg as rspec

# Create a Request object
request = portal.context.makeRequestRSpec()

# Add a single raw PC (physical machine)
node = request.RawPC("node")

# Specify the OS image (Ubuntu 22.04 is the default for this node)
node.disk_image = "urn:publicid:IDN+emulab.net+image+emulab-ops//UBUNTU24-64-STD"

# --- Add these two lines ---
# 1. Request the specific 'c220g5' hardware type
node.hardware_type = "c220g5"

# 2. Pin the experiment to the Wisconsin cluster
node.Site("wisc")
# --- End of new lines ---

# Add an "execute service" to run your setup script
node.addService(rspec.Execute(shell="bash", 
                              command="/local/repository/setup.sh"))

# Print the RSpec
portal.context.printRequestRSpec()