
# MAIN FUNCTION TO CALL THE L1C MODULE

from l1c.src.l1c import l1c

# Directory - this is the common directory for the execution of the E2E, all modules
#auxdir = r'C:\\Users\\pleon\\EODP_files\\EODP\\auxiliary'
# GM dir + L1B dir
#indir = r'C:\\Users\\pleon\\EODP_files\\SHARED\\EODP-TS-E2E\\gm_out,C:\\Users\\pleon\\EODP_files\\SHARED\\EODP-TS-E2E\\output_2' # small scene
#outdir = r'C:\\Users\\pleon\\EODP_files\\SHARED\\EODP-TS-E2E\\output_3'

auxdir = r'C:\\Users\\pleon\\EODP_files\\EODP\\auxiliary'
indir = r'C:\\Users\\pleon\\EODP_files\\SHARED\\EODP_TER_2021\\EODP-TS-L1C\\input\\gm_alt100_act_150,C:\\Users\\pleon\\EODP_files\\SHARED\\EODP_TER_2021\\EODP-TS-L1C\\input\\l1b_output' # small scene
outdir = r'C:\\Users\\pleon\\EODP_files\\SHARED\\EODP_TER_2021\\EODP-TS-L1C\\Myoutputs'

# Initialise the ISM
myL1c = l1c(auxdir, indir, outdir)
myL1c.processModule()
