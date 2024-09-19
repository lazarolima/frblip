# Necessary libraries.
import numpy as np
import math as math
from frblip import FastRadioBursts, RadioTelescope
from astropy import units as u
import pymp
import random

# BEGIN OF INPUTS
# Input file names
input_main_name   = 'bingo'
input_array_name  = 'mirror_4m'
input_altaz_file = '/home/llima/frblip/Script_Gabriel/outrigger_3direc.dat'

# Output file names
output_counter    = '/home/llima/frblip/Script_Gabriel/counter_frb.dat'
output_details    = '/home/llima/frblip/Script_Gabriel/frb_details.dat'

# Number of days and cpus
cpus = 20
days = 30

# Beams information (except BINGO beams) 
num_stations  = 1
directivity4  = 32.21 * u.Unit('dB(1/sr)') # 4m mirror/dipole
directivity5  = 34.15 * u.Unit('dB(1/sr)') # 5m mirror/dipole
directivity6  = 35.89 * u.Unit('dB(1/sr)') # 6m mirror/dipole
directivity30 = 48.88 * u.Unit('dB(1/sr)') # 30m dipole
directivity40 = 51.38 * u.Unit('dB(1/sr)') # 40m dipole

beams_directivity = directivity4 # <-------------- CHOOSE DIRECTIVITY HERE!!!
  
# Detection criteria
minimum_snr                   = 1.5
detection_snr                 = 5.0
localization_total_snr        = 3.0
localization_per_baseline_snr = 1.0

# Cosmology and observarion
spectral_index = 0.0     # Spectral Index distribution.
zmax           = 6.0     # z max for FRB in cosmological FRBs
dec            = (-90.0,90.0) # Survey dec

# DM_host model
ETG        = ('ETG')
LHEG       = ('LHEG') 
LTG_NE2001 = ('LTG', 'NE2001')
LTG_YMW16  = ('LTG', 'YMW16')
ALG_NE2001 = ('LTG', 'NE2001')
ALG_YMW16  = ('LTG', 'YMW16')
LOGNORMAL  = ('LOGNORMAL')
GAUSSIAN   = ('GAUSSIAN')

host_model = ALG_YMW16

if (host_model == GAUSSIAN):
    host_dist = 'normal'
else:
    host_dist = 'lognormal'
# END OF INPUTS

#CREATE THE TELESCOPES DICTIONARY AND SETUP THE POSITIONS AND OFFSETS
telescopes = {}
altaz = np.loadtxt(input_altaz_file)

telescopes['MAIN'] = RadioTelescope(input_main_name, system_temperature=70*u.K)
telescopes['ARRAY'] = RadioTelescope(input_array_name, system_temperature=70*u.K, alt=altaz[:,0]*u.deg, az=altaz[:,1]*u.deg)

# SETUP THE INTF KEYS AND THE AMOUNT OF INTF
observ_keys   = ['MAIN', 'ARRAY', 'INTF_ARRAY', 'INTF_MAIN_ARRAY']
inter_keys    = ['INTF_ARRAY', 'INTF_MAIN_ARRAY']
inter_indices = [2, 3]

total_observ = len(observ_keys)

#multi_intf_array = (np.math.factorial(num_stations) / (np.math.factorial(num_stations - 2) * 2))
multi_intf_array = 1

# COUNTERS STARTERS
num_candidates     = 0
num_detection      = 0
num_localization_1 = 0
count_localization = 0
num_localization_2 = 0
main_contribution  = 0

main_location = telescopes['MAIN'].location 

# COUNTING AND MOCK GENERATOR ALGORITHM

candidates_matrix = pymp.shared.list()  

with pymp.Parallel(cpus) as P:
    for i in P.range(days):
        candidates = []
        frb_candidates = []
        candidates_matrix.append([[],[],[],[],[],[],[],[],[],[],[],[]]) #Day, No, redshift, logL, alt, az, DM, candidate?, detected?, in 1baseline?, in 2 baselines?, by BINGO?
        mock = FastRadioBursts(verbose = True, spectral_index = spectral_index, zmax = zmax, host_model = host_model, host_dist = host_dist, dec=dec)
        altaz = mock.altaz(main_location)
        mock.observe(telescopes, altaz = altaz, verbose = False)
        mock.interferometry('MAIN', 'ARRAY')
        mock.interferometry('ARRAY')

        snr = mock.signal_to_noise(observ_keys)

        for j in range(total_observ):
            where = np.unique(np.where(snr[observ_keys[j]] > minimum_snr)[0])
            if (np.array(where).size > 0):
                for k in range(np.array(where).size):
                    frb_candidates.append(where[k])
        
        num_candidates_day = len(list(set(frb_candidates)))
        if (num_candidates_day == 0):
            continue
                        
        num_candidates += num_candidates_day
        for k in range(num_candidates_day):
            candidates_matrix[i][0].append(i)
            candidates_matrix[i][1].append(list(set(frb_candidates))[k])
            candidates_matrix[i][2].append(mock.redshift[list(set(frb_candidates))[k]])
            candidates_matrix[i][3].append(mock.log_luminosity[list(set(frb_candidates))[k]].value)
            candidates_matrix[i][4].append(mock['MAIN'].altaz.alt.value[list(set(frb_candidates))[k]])
            candidates_matrix[i][5].append(mock['MAIN'].altaz.az.value[list(set(frb_candidates))[k]])
            candidates_matrix[i][6].append(mock.dispersion_measure[list(set(frb_candidates))[k]].value)
            candidates_matrix[i][7].append(1)
            
        del(mock)
        sum_observ = np.zeros(num_candidates_day)
    
        for j in range(num_candidates_day):
            for k in range(total_observ):
                if (k == 0):
                    sum_observ[j] += (np.array(snr[observ_keys[k]])[list(set(frb_candidates))[j]]**2).sum(0)
                elif (k == 1):
                    sum_observ[j] += num_stations * (np.array(snr[observ_keys[k]])[list(set(frb_candidates))[j]]**2).sum(0)
                elif (k == 2):
                    sum_observ[j] += multi_intf_array * (np.array(snr[observ_keys[k]])[list(set(frb_candidates))[j]]**2).sum(0)
                else:
                    for l in range (28):
                        sum_observ[j] += num_stations * (np.array(snr[observ_keys[k]])[list(set(frb_candidates))[j],l]**2).sum(0)

            if (np.sqrt(sum_observ[j]) > detection_snr):
                num_detection += 1
                candidates_matrix[i][8].append(1)
            else:
                candidates_matrix[i][8].append(0)

        sum_intf = np.zeros(num_candidates_day)

        for j in range(num_candidates_day):
            for k in inter_indices:
                if (k == 2):
                    sum_intf[j] += multi_intf_array * (np.array(snr[observ_keys[k]])[list(set(frb_candidates))[j]]**2).sum(0)
                    count_localization += ((np.array(snr[observ_keys[k]])[list(set(frb_candidates))[j]]) > localization_per_baseline_snr).sum()
                else:
                    for l in range (28):
                        sum_intf[j] += num_stations * (np.array(snr[observ_keys[k]])[list(set(frb_candidates))[j],l]**2).sum(0)
                        count_localization += ((np.array(snr[observ_keys[k]])[list(set(frb_candidates))[j],l]) > localization_per_baseline_snr).sum()
                        if (((np.array(snr[observ_keys[k]])[list(set(frb_candidates))[j],l]) > localization_per_baseline_snr).sum() >= 1):
                            main_contribution += 1

            if (np.sqrt(sum_intf[j]) > localization_total_snr) and (np.sqrt(sum_observ[j]) > detection_snr) and (count_localization >= 1):
                num_localization_1 += 1
                candidates_matrix[i][9].append(1)
            else:
                candidates_matrix[i][9].append(0)

            if (np.sqrt(sum_intf[j]) > localization_total_snr) and (np.sqrt(sum_observ[j]) > detection_snr) and (count_localization >= 2):
                num_localization_2 += 1
                candidates_matrix[i][10].append(1)
            else:
                candidates_matrix[i][10].append(0)

            if (np.sqrt(sum_intf[j]) > localization_total_snr) and (np.sqrt(sum_observ[j]) > detection_snr) and (count_localization >= 1) and (main_contribution >= 1):
                candidates_matrix[i][11].append(1)
            else:
                candidates_matrix[i][11].append(0)

            count_localization = 0
            main_contribution = 0

        del(candidates)
        del(frb_candidates)
        del(snr)

# OUTPUTS
with open(output_counter, "w") as text_file_counts:
    text_file_counts.write("")
    print('Number of candidates: ' + str(len(candidates_matrix)) + '\n')
    text_file_counts.write(str(days) + "\t" + str(len(candidates_matrix)) + "\n")

with open(output_details, "w") as text_file_frb:
    text_file_frb.write("")
    for day_matrix in candidates_matrix:
        for k in range(len(day_matrix[0])):
            text_file_frb.write(str(day_matrix[0][k]) + "\t" + str(day_matrix[1][k]) + "\t" +
                                str(day_matrix[2][k]) + "\t" + str(day_matrix[3][k]) + "\t" +
                                str(day_matrix[4][k]) + "\t" + str(day_matrix[5][k]) + "\t" +
                                str(day_matrix[6][k]) + "\t" + str(day_matrix[7][k]) + "\t" +
                                str(day_matrix[8][k]) + "\t" + str(day_matrix[9][k]) + "\t" +
                                str(day_matrix[10][k]) + "\t" + str(day_matrix[11][k]) + "\n")


# NOW WE JUST HAVE TO WRITE THE RESULTS 
with open(output_counter, "w") as text_file_counts:
    text_file_counts.write("")
    print('Number of candidates: ' + str(num_candidates) + '\n')
    print('Number of detection: ' + str(num_detection) + '\n')
    print('Number of localization in at least 1 point: ' + str(num_localization_1) + '\n')
    print('Number of localization in at least 2 points: ' + str(num_localization_2) + '\n')
    text_file_counts.write(str(days) + "\t" + str(num_candidates) + "\t" + str(num_detection) +  "\t" +
                           str(num_localization_1) + "\t" + str(num_localization_2) + "\n")

size = len(candidates_matrix)
count = 0

for i in range(size):
    if (candidates_matrix[i]==[[], [], [], [], [], [], [], [], [], [], [], []]):
        count += 1

for i in range(count):
    candidates_matrix.remove([[], [], [], [], [], [], [], [], [], [], [], []])

with open(output_details, "w") as text_file_frb:
    text_file_frb.write("")
    print(candidates_matrix)
    for i in range(len(candidates_matrix)):
        for k in range(len(candidates_matrix[i][0])):
            print(candidates_matrix[i][1][k])
            text_file_frb.write(str(candidates_matrix[i][0][k]) + "\t" + str(candidates_matrix[i][1][k]) + "\t" +
                                str(candidates_matrix[i][2][k]) + "\t" + str(candidates_matrix[i][3][k]) + "\t" +
                                str(candidates_matrix[i][4][k]) + "\t" + str(candidates_matrix[i][5][k]) + "\t" +
                                str(candidates_matrix[i][6][k]) + "\t" + str(candidates_matrix[i][7][k]) + "\t" +
                                str(candidates_matrix[i][8][k]) + "\t" + str(candidates_matrix[i][9][k]) + "\t" +
                                str(candidates_matrix[i][10][k]) + "\t" + str(candidates_matrix[i][11][k]) + "\n")
                                
