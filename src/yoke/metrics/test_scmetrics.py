####################################################
# This program computes shaped-charge metrics from
# npz files from Pagosa simiulations.
#
# This program uses the SCMetrics class to compute
# the shaped-charge metrics. The class is imported
# and used by this "main" script.
#
# This program serves as an example of how to
# call and use the SCMetrics class.
#
# Created by Derek Armstrong, XCP-8, June 2024.
####################################################

import os
import numpy as np
from shaped_charge_metrics import SCmetrics

if __name__ == '__main__':

  # Set the directory that contains the npz files.
  # CHANGE THIS BASED ON LOCATION OF DATA!
  data_dir = "../../../npz/lsc240420/"
  # file_name_format used with this extension ... % (runID,time_index)
  file_name_format = "lsc240420_id%05d_pvi_idx%05d.npz"
  # set output file name, it will be csv format
  outfile = "sc_metrics.csv"

  # Set the run IDs to compute metrics for
  # runID_list= [i+1 for i in range(100)]
  # runID_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
  #               11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
  #               21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
  #               31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
  #                   42, 43, 44, 45, 46, 47, 48, 49, 50,
  #               51, 52, 53, 54, 55, 56, 57, 58, 59, 60,
  #               61, 62, 63, 64, 65, 66, 67, 68, 69, 70,
  #               71, 72, 73, 74, 75, 76, 77, 78, 79, 80,
  #               81, 82, 83, 84, 85, 86, 87, 88, 89, 90,
  #               91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101]
  runID_list = [99, 92, 33, 18, 94, 34]

  # Set the simulation times indexes to compute metrics for
  time_list = [100]

  # set number of files to compute metrics for
  nruns = len(runID_list)
  ntimes = len(time_list)

  # initialize arrays holding metrics
  # this is not necessary if only writing results to file
  # this will change, naturally, based on desired metrics
  eff_jet_mass = np.zeros((nruns, ntimes))
  jet_mass = np.zeros((nruns, ntimes))
  HE_mass = np.zeros((nruns, ntimes))
  jet_width_avg = np.zeros((nruns, ntimes))
  jet_width_std = np.zeros((nruns, ntimes))
  jet_width_max = np.zeros((nruns, ntimes))
  KE = np.zeros((nruns, ntimes))
  sqrt_KE = np.zeros((nruns, ntimes))

  # write header line for csf file
  fid = open(outfile, "w")
  fid.write("key, time_index, eff_jet_mass_percent, jet_mass, HE_mass, avg_width_1, std_width_1, max_width_1, KE_2D, sqrt_KE_2D\n")

  # loop through the runIDs and time indeces
  # and compute the metrics
  for irun, runID in enumerate(runID_list):
    for itime, time_index in enumerate(time_list):
      npzfile = data_dir + file_name_format % (runID, time_index)
      metrics = SCmetrics(npzfile)
      eff_jet_mass[irun, itime] = metrics.get_eff_jet_mass(asPercent=True)
      jet_mass[irun, itime] = metrics.get_jet_mass()
      HE_mass[irun, itime] = metrics.get_HE_mass()
      avg, stdj, maxj = metrics.get_jet_width_stats(vel_thres=0.1)
      jet_width_avg[irun, itime] = avg
      jet_width_std[irun, itime] = stdj
      jet_width_max[irun, itime] = maxj
      KE[irun, itime] = metrics.get_jet_rho_velsq_2D()
      sqrt_KE[irun, itime] = metrics.get_jet_sqrt_rho_vel_2D()
      ifile = os.path.basename(npzfile)
      fid.write("%s, %i, %f, %f, %f, %f, %f, %f, %f, %f\n" % (ifile, time_index, eff_jet_mass[irun, itime], jet_mass[irun, itime], HE_mass[irun, itime],
                                                              jet_width_avg[irun, itime], jet_width_std[irun, itime], jet_width_max[irun, itime],
                                                              KE[irun, itime], sqrt_KE[irun, itime]))

  fid.close()
