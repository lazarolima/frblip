'''FRB Mock generator (Phased Arrays).
Author: Gabriel Hoerning, Msc student, University of São Paulo, 2022.
Based on code: FRBlip. Author: Marcelo Vieira Santos, University of Campina Grande, 2022.
BINGO Telescope Collaboration'''

'''This script generates a Fast Radio Burst (FRB) mock for he localiztion of them, based on
a cosmology (which distributes the FRBs over the sky) and a telescope frequency bands.
In this version, the script is ready to generate a FRB mock for the BINGO Telescope equipped
with X numbers of Phased Arrays with Y beams per Phased Array.'''

'''To run the code you will need to be inputing:
	Beams physical details file.npz (can be changed during the code) for BINGO/PhasedArrays beams.
	Beams positions file.npz (can be changed during the code) for BINGO/PhasedArrays beams.
	Number of days.
	Output file names'''
	
'''HOW TO RUN: 
	Once the information above is given, you can run the code by using the following
	command in a terminal window:
	
	$ python3 INSERT_NAME_HERE_GABRIEL!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!.py
	
WARNING: The FRB Mock generator for Phased Arrays consumes a lot of Memory due to the the number
	 of interferometries that the script needs to do. By running 1826 days of data (equivalent
	 to 5 years) it's very likely that the code will crash, reset and then you will lose all the
	 run. 
	 
	 I STRONGLY RECOMMEND YOU TO RUN THIS CODE FOLLOWING THIS PROCEDURE:
	 
	 1st: Choose a number of days which you know your machine memory can deal with. Ex: 40 days.
	 2nd: Suppose you want to run 1826 days. Then 45(runs) * 40(days) + 26(days) = 1826(days).
	      So my suggestion is to you run the script 45 times for 40 days and just 1 time for 26 
	      days.
	 3rd: You can easily run the code 45 times as with the following command line:
	      
	      $ for i in {1..45} do python3 INSERT_NAME_HERE_GABRIEL!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!.py; done
	      
	      Then you just need to run 1 more time with 26 days (with the normal command line of python)
	      to get 1826 days.
	      
	 P.S.: The code won't rewrite any file, so you don't need to worry about that.'''

# Necessary libraries.
import numpy as np
#from tqdm import trange
import math as math
from frblip import FastRadioBursts, RadioTelescope
from astropy import units as u
from astropy import coordinates as co
import random
import pymp

# BEGIN OF INPUTS
# Input file names
input_main_name   = 'bingo'
input_array_name  = 'naked_horn'
input_altaz_file = '/home/lazarolima/Script_Gabriel/outrigger_altaz.dat'

# Output file names
output_counter    = '/home/lazarolima/Script_Gabriel/counter_frb.dat'
output_details    = '/home/lazarolima/Script_Gabriel/frb_details.dat'

# Number of days and cpus
cpus = 6
days = 5

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
zmax           = 10    # z max for FRB in cosmological FRBs
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
main = RadioTelescope(input_main_name)
#altaz = np.loadtxt(input_altaz_file)

telescopes['MAIN'] = RadioTelescope(input_main_name, system_temperature=70*u.K)
#telescopes['ARRAY'] = RadioTelescope(input_array_name, alt=altaz[:,0]*u.deg,az=altaz[:,1]*u.deg, directivity = beams_directivity, location = main.location)
telescopes['ARRAY'] = RadioTelescope(input_array_name, system_temperature=70*u.K)

# SETUP THE INTF KEYS AND THE AMOUNT OF INTF
observ_keys   = ['MAIN', 'ARRAY', 'INTF_MAIN_ARRAY']
inter_keys    = ['INTF_MAIN_ARRAY']
inter_indices = [1,2]

total_observ = len(observ_keys)

# SETUP SEED    
counter_file = np.loadtxt(output_counter)
if counter_file.T[0].size == 0:
    seed_starter = 0
else:
    seed_starter = counter_file.T[0].sum(0)
    
print('The seed starter of this run is ' + str(int(seed_starter)))

# MULTIPLIERS SETUP
# We have to increse some SNR by a certain amount because we are not running all
# the observation (to save computationl resources). To compensate this, we have 
# to multiply some SNR by a certain amount based on the number of stations and beams.

#multi_intf_array = (np.math.factorial(num_stations) / (np.math.factorial(num_stations - 2) * 2))
multi_intf_array = 1

# COUNTERS STARTERS
num_candidates     = 0
num_detection      = 0
num_localization_1 = 0
count_localization = 0
num_localization_2 = 0
main_contribution  = 0
#candidates_matrix  = pymp.shared.list()
candidates_matrix  = []

main_location = telescopes['MAIN'].location

# COUNTING AND MOCK GENERATOR ALGORITHM
# DO NOT EVER TOUCH IN ANYTHING HERE UNLESS YOU KKOW EXCATLY WHAT YOU ARE DOING. 
# THIS IS THE HEART OF THE COUNTING.
with pymp.Parallel(cpus) as P:
    for i in P.range(days):
        candidates = []
        frb_candidates = []
        candidates_matrix.append([[],[],[],[],[],[],[],[],[],[],[],[]]) #Day, No, redshift, logL, alt, az, DM, candidate?, detected?, in 1baseline?, in 2 baselines?, by BINGO?
        #np.random.seed(i + int(seed_starter))
        np.random.seed(0)
        mock = FastRadioBursts(random_state=42, duration=days, verbose = True, spectral_index = spectral_index, zmax = zmax, host_model = host_model, host_dist = host_dist, dec_range = dec)
        #altaz = mock.altaz(main_location)
        #mock.observe(telescopes, altaz = altaz, verbose = False)
        mock.observe(telescopes, verbose = False, location=main_location)
        mock.interferometry('MAIN', 'ARRAY', overwrite=True)
        #mock.interferometry('ARRAY')


        """snr = mock.signal_to_noise(observ_keys)

        for j in range(total_observ):
            where = np.unique(np.where(snr[observ_keys[j]] > minimum_snr)[0])
            if (np.array(where).size > 0):
                for k in range(np.array(where).size):
                    frb_candidates.append(where[k])"""
        
        # SNR Calculation
        snr = {}
        for key in observ_keys:
            snr[key] = mock.signal_to_noise(key, todense=True, total=True)

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
                key = observ_keys[k]
                snr_array = np.array(snr[key])
                candidate_index = list(set(frb_candidates))[j]

                if k == 2:
                    if candidate_index < len(snr_array):
                        sum_intf[j] += multi_intf_array * (snr_array[candidate_index] ** 2).sum(0)
                        count_localization += (snr_array[candidate_index] > localization_per_baseline_snr).sum()
                    else:
                        print(f"Candidate index {candidate_index} out of range for snr_array")
                else:
                    if candidate_index < len(snr_array):
                        sum_intf[j] += num_stations * (snr_array[candidate_index] ** 2).sum(0)
                        count_localization += (snr_array[candidate_index] > localization_per_baseline_snr).sum()
                        if (snr_array[candidate_index] > localization_per_baseline_snr).sum() >= 1:
                            main_contribution += 1
                    else:
                        print(f"Candidate index {candidate_index} out of range for snr_array")



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

# NOW WE JUST HAVE TO WRITE THE RESULTS 
with open(output_counter, "a") as text_file_counts:
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

with open(output_details, "a") as text_file_frb:
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
                                
