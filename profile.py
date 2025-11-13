"""A profile to run the CSC2235 project.

Instructions:
Wait for the node to be "Ready". This means the setup script
has finished installing pip and all Python requirements.
Your project code is in /local/repository. To run all benchmarks, execute:
    sh /local/repository/run_benchmarks.sh
To run a specific benchmark:
    sh /local/repository/run_benchmarks.sh <benchmark_name>
To run a specific benchmark with a specific framework:
    sh /local/repository/run_benchmarks.sh <benchmark_name> <framework_name>
For example, to run the NYC Taxi benchmark with DuckDB, execute:
    sh /local/repository/run_benchmarks.sh nyc_taxi duckdb
"""

import geni.portal as portal
import geni.rspec.pg as rspec

portal.context.defineParameter(
    "hwtype", "Physical Node Type",
    portal.ParameterType.STRING, "c220g5",
    [('c220g5', 'Wisconsin: c220g5 (Intel Skylake)'),
     ('c240g5', 'Wisconsin: c240g5 (Intel Skylake)')],
    "Select an x86-based physical node type. If your first choice is unavailable, try another."
)

params = portal.context.bindParameters()

request = portal.context.makeRequestRSpec()

node = request.RawPC("node")

node.disk_image = "urn:publicid:IDN+emulab.net+image+emulab-ops//UBUNTU24-64-STD"

node.hardware_type = params.hwtype

node.addService(rspec.Execute(shell="bash", 
                              command="/local/repository/setup.sh"))

portal.context.printRequestRSpec()