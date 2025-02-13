import numpy as np
import dynamics as dyn
import config_opc
import plot_utils as pu
import simulate as simu
import optimal as opt
from matplotlib import pyplot as plt
import time
import os
import argparse

if __name__ == '__main__':
     parser = argparse.ArgumentParser(description='Compare open loop results.')
     parser.add_argument('--from_file_n', action='store_true', help='Load non-morphing data from file')
     parser.add_argument('--file_folder_n', type=str, default="pics/results_open_20250128-1659", help='Folder for non-morphing data')
     parser.add_argument('--from_file_m', action='store_true', help='Load morphing data from file')
     parser.add_argument('--file_folder_m', type=str, default="pics/results_open_20250131-1407_1", help='Folder for morphing data')
     parser.add_argument('--from_file_n_f', action='store_true', help='Load non-morphing fuel data from file')
     parser.add_argument('--file_folder_n_f', type=str, default="pics/results_open_20250128-1659", help='Folder for non-morphing fuel data')
     parser.add_argument('--from_file_m_f', action='store_true', help='Load morphing fuel data from file')
     parser.add_argument('--file_folder_m_f', type=str, default="pics/results_open_20250131-1701", help='Folder for morphing fuel data')
     parser.add_argument('--from_file_n_m', action='store_true', help='Load non-morphing fuel data from file')
     parser.add_argument('--file_folder_n_m', type=str, default="pics/results_open_20250210-0038_0", help='Folder for non-morphing fuel data')
     parser.add_argument('--from_file_m_m', action='store_true', help='Load morphing fuel data from file')
     parser.add_argument('--file_folder_m_m', type=str, default="pics/results_open_20250210-0038_0", help='Folder for morphing fuel data')
     parser.add_argument('--from_file_n_b', action='store_true', help='Load non-morphing fuel data from file')
     parser.add_argument('--file_folder_n_b', type=str, default="pics/results_open_20250210-0038_7", help='Folder for non-morphing fuel data')
     parser.add_argument('--from_file_m_b', action='store_true', help='Load morphing fuel data from file')
     parser.add_argument('--file_folder_m_b', type=str, default="pics/results_open_20250210-0038_7", help='Folder for morphing fuel data')
     parser.add_argument('--K', type=int, default=1, help='PID gain numbers')
     parser.add_argument('--shown', action='store_true', help='Show the plot')
     args = parser.parse_args()

     from_file_n = args.from_file_n
     file_folder_n = args.file_folder_n
     from_file_m = args.from_file_m
     file_folder_m = args.file_folder_m
     from_file_n_f = args.from_file_n_f
     file_folder_n_f = args.file_folder_n_f
     from_file_m_f = args.from_file_m_f
     file_folder_m_f = args.file_folder_m_f
     from_file_n_m = args.from_file_n_m
     file_folder_n_m = args.file_folder_n_m
     from_file_m_m = args.from_file_m_m
     file_folder_m_m = args.file_folder_m_m
     from_file_n_b = args.from_file_n_b
     file_folder_n_b = args.file_folder_n_b
     from_file_m_b = args.from_file_m_b
     file_folder_m_b = args.file_folder_m_b
     K = args.K
     shown = args.shown

     cur_time = time.strftime("%Y%m%d-%H%M", time.localtime())
     pic_folder = "pics\\results_open_"+cur_time
     if not os.path.exists(pic_folder):
          os.mkdir(pic_folder)
     switch_time = 0.5
     constant_height=300
     tra_ref = simu.generate_ref_trajectory_varying(constant_height=constant_height, switch_time=switch_time, high_height=350, low_height=250)
     np.savez(f'{pic_folder}\\h_ref.npz', h_r = tra_ref['h_r_seq'])

     data_init_path = "pics/results_open_20250127-1745/data_morphing.npz"
     with np.load(data_init_path) as data_init:
               u_init_given = data_init['u_m']

     for k in range(K):
          # file_folder_n = f"pics\\results_open_20250131-1407_{k}"
          # file_folder_m = f"pics\\results_open_20250131-1407_{k}"
          # file_folder_n_f = f"pics\\results_open_20250131-1407_{k}"
          # file_folder_m_f = f"pics\\results_open_20250131-1407_{k}"

          pic_folder = f"pics\\results_open_{cur_time}_{k}"
          if not os.path.exists(pic_folder):
               os.mkdir(pic_folder)
          config_opc.PARA_KP = config_opc.PARA_KP_L + k * (config_opc.PARA_KP_U - config_opc.PARA_KP_L) / (K - 1) if K > 1 else config_opc.PARA_KP
          with open(f'{pic_folder}\\config_kp.txt', 'w') as f:
               f.write(f'config_opc.PARA_KP = {config_opc.PARA_KP}\n')
          print(f'k = {k}')
          print(f'config_opc.PARA_KP = {config_opc.PARA_KP}')         
          
          # Non-Morphing, major cost
          if from_file_n:
               # Load
               with np.load(f'{file_folder_n}\\data_nomorphing.npz') as data_nomorphing:
                    keys = ['x_n', 'y_n', 'z_n', 'u_n', 'j_f_n', 'aero_info_n']
                    results_n = [data_nomorphing[key] for key in keys]
          else:
               # x_optimal, y_optimal, z_optimal, u_optimal, j_optimal = opt.generate_PS_solution_scaled(x0=config_opc.PARA_X0,   trajectory_ref=tra_ref, morphing_disabled=0.5, maxiter=500, init_method='given', given_input=u_init_given)
               x_optimal, y_optimal, z_optimal, u_optimal, j_optimal = opt.generate_PS_solution_scaled(x0=config_opc.PARA_X0,     trajectory_ref=tra_ref, morphing_disabled=0.5, maxiter=500)
               t, x, y, z, u_n = opt.interpolate_optimal_trajectory(x_optimal, y_optimal, z_optimal, u_optimal, j_optimal,   t_switch=tra_ref["t_switch"], from_scaled=True)
               results_n = simu.simulate_auxiliary(x0=config_opc.PARA_X0, trajectory_ref=tra_ref, control_method="given",    given_input=u_n)
               # Save
               x_n, y_n, z_n, u_n, j_f_n, aero_info_n = results_n
               np.savez(f'{pic_folder}\\data_nomorphing.npz', x_n=x_n, y_n=y_n, z_n=z_n, u_n=u_n, j_f_n=j_f_n,
                      aero_info_n=aero_info_n)

          # Morphing, major cost
          if from_file_m:
               # Load
               with np.load(f'{file_folder_m}\\data_morphing.npz') as data_morphing:
                    keys = ['x_m', 'y_m', 'z_m', 'u_m', 'j_f_m', 'aero_info_m']
                    results_m = [data_morphing[key] for key in keys]
          else:
               x_optimal, y_optimal, z_optimal, u_optimal, j_optimal = opt.generate_PS_solution_scaled(x0=config_opc.PARA_X0,     trajectory_ref=tra_ref, morphing_disabled=None)
               t, x, y, z, u_m = opt.interpolate_optimal_trajectory(x_optimal, y_optimal, z_optimal, u_optimal, j_optimal,   t_switch=tra_ref["t_switch"], from_scaled=True)
               results_m = simu.simulate_auxiliary(x0=config_opc.PARA_X0, trajectory_ref=tra_ref, control_method="given",    given_input=u_m)
               # Save
               x_m, y_m, z_m, u_m, j_f_m, aero_info_m = results_m
               np.savez(f'{pic_folder}\\data_morphing.npz', x_m=x_m, y_m=y_m, z_m=z_m, u_m=u_m, j_f_m=j_f_m,
                      aero_info_m=aero_info_m)

          # Non-Morphing, major cost + fuel cost
          if from_file_n_f:
               # Load
               with np.load(f'{file_folder_n_f}\\data_nomorphing_fuel.npz') as data_nomorphing_fuel:
                    keys = ['x_n_f', 'y_n_f', 'z_n_f', 'u_n_f', 'j_f_n_f', 'aero_info_n_f']
                    results_n_f = [data_nomorphing_fuel[key] for key in keys]
          else:
               x_optimal, y_optimal, z_optimal, u_optimal, j_optimal = opt.generate_PS_solution_scaled(x0=config_opc.PARA_X0,     trajectory_ref=tra_ref, morphing_disabled=0.5, fun_obj=opt.function_objective_fuel, maxiter=500)
               t, x, y, z, u_n_f = opt.interpolate_optimal_trajectory(x_optimal, y_optimal, z_optimal, u_optimal, j_optimal, t_switch=tra_ref["t_switch"], from_scaled=True)
               results_n_f = simu.simulate_auxiliary(x0=config_opc.PARA_X0, trajectory_ref=tra_ref, control_method="given", given_input=u_n_f)
               # Save
               x_n_f, y_n_f, z_n_f, u_n_f, j_f_n_f, aero_info_n_f = results_n_f
               np.savez(f'{pic_folder}\\data_nomorphing_fuel.npz', x_n_f=x_n_f, y_n_f=y_n_f, z_n_f=z_n_f, u_n_f=u_n_f, j_f_n_f=j_f_n_f, aero_info_n_f=aero_info_n_f)

          # Morphing, major cost + fuel cost
          if from_file_m_f:
               # Load
               with np.load(f'{file_folder_m_f}\\data_morphing_fuel.npz') as data_morphing_fuel:
                    keys = ['x_m_f', 'y_m_f', 'z_m_f', 'u_m_f', 'j_f_m_f', 'aero_info_m_f']
                    results_m_f = [data_morphing_fuel[key] for key in keys]
          else:
               x_optimal, y_optimal, z_optimal, u_optimal, j_optimal = opt.generate_PS_solution_scaled(x0=config_opc.PARA_X0,     trajectory_ref=tra_ref, morphing_disabled=None, fun_obj=opt.function_objective_fuel)
               t, x, y, z, u_m_f = opt.interpolate_optimal_trajectory(x_optimal, y_optimal, z_optimal, u_optimal, j_optimal, t_switch=tra_ref["t_switch"], from_scaled=True)
               results_m_f = simu.simulate_auxiliary(x0=config_opc.PARA_X0, trajectory_ref=tra_ref, control_method="given", given_input=u_m_f)
               # Save
               x_m_f, y_m_f, z_m_f, u_m_f, j_f_m_f, aero_info_m_f = results_m_f
               np.savez(f'{pic_folder}\\data_morphing_fuel.npz', x_m_f=x_m_f, y_m_f=y_m_f, z_m_f=z_m_f, u_m_f=u_m_f, j_f_m_f=j_f_m_f, aero_info_m_f=aero_info_m_f)
          
          # Non-Morphing, major cost + manu cost
          if from_file_n_m:
               # Load
               with np.load(f'{file_folder_n_m}\\data_nomorphing_manu.npz') as data_nomorphing_manu:
                    keys = ['x_n_m', 'y_n_m', 'z_n_m', 'u_n_m', 'j_f_n_m', 'aero_info_n_m']
                    results_n_m = [data_nomorphing_manu[key] for key in keys]
          else:
               x_optimal, y_optimal, z_optimal, u_optimal, j_optimal = opt.generate_PS_solution_scaled(x0=config_opc.PARA_X0, trajectory_ref=tra_ref, morphing_disabled=0.5, fun_obj=opt.function_objective_manu, maxiter=500)
               t, x, y, z, u_n_m = opt.interpolate_optimal_trajectory(x_optimal, y_optimal, z_optimal, u_optimal, j_optimal, t_switch=tra_ref["t_switch"], from_scaled=True)
               results_n_m = simu.simulate_auxiliary(x0=config_opc.PARA_X0, trajectory_ref=tra_ref, control_method="given", given_input=u_n_m)
               # Save
               x_n_m, y_n_m, z_n_m, u_n_m, j_f_n_m, aero_info_n_m = results_n_m
               np.savez(f'{pic_folder}\\data_nomorphing_manu.npz', x_n_m=x_n_m, y_n_m=y_n_m, z_n_m=z_n_m, u_n_m=u_n_m, j_f_n_m=j_f_n_m, aero_info_n_m=aero_info_n_m)

          # Morphing, major cost + manu cost
          if from_file_m_m:
               # Load
               with np.load(f'{file_folder_m_m}\\data_morphing_manu.npz') as data_morphing_manu:
                    keys = ['x_m_m', 'y_m_m', 'z_m_m', 'u_m_m', 'j_f_m_m', 'aero_info_m_m']
                    results_m_m = [data_morphing_manu[key] for key in keys]
          else:
               x_optimal, y_optimal, z_optimal, u_optimal, j_optimal = opt.generate_PS_solution_scaled(x0=config_opc.PARA_X0, trajectory_ref=tra_ref, morphing_disabled=None, fun_obj=opt.function_objective_manu)
               t, x, y, z, u_m_m = opt.interpolate_optimal_trajectory(x_optimal, y_optimal, z_optimal, u_optimal, j_optimal, t_switch=tra_ref["t_switch"], from_scaled=True)
               results_m_m = simu.simulate_auxiliary(x0=config_opc.PARA_X0, trajectory_ref=tra_ref, control_method="given", given_input=u_m_m)
               # Save
               x_m_m, y_m_m, z_m_m, u_m_m, j_f_m_m, aero_info_m_m = results_m_m
               np.savez(f'{pic_folder}\\data_morphing_manu.npz', x_m_m=x_m_m, y_m_m=y_m_m, z_m_m=z_m_m, u_m_m=u_m_m, j_f_m_m=j_f_m_m, aero_info_m_m=aero_info_m_m)

            # Non-Morphing, major cost + fuel cost + manu cost
          if from_file_n_b:
                # Load
               with np.load(f'{file_folder_n_b}\\data_nomorphing_both.npz') as data_nomorphing_both:
                    keys = ['x_n_b', 'y_n_b', 'z_n_b', 'u_n_b', 'j_f_n_b', 'aero_info_n_b']
                    results_n_b = [data_nomorphing_both[key] for key in keys]
          else:
               x_optimal, y_optimal, z_optimal, u_optimal, j_optimal = opt.generate_PS_solution_scaled(x0=config_opc.PARA_X0,trajectory_ref=tra_ref, morphing_disabled=0.5, fun_obj=opt.function_objective_both, maxiter=500)
               t, x, y, z, u_n_b = opt.interpolate_optimal_trajectory(x_optimal, y_optimal, z_optimal, u_optimal, j_optimal,t_switch=tra_ref["t_switch"], from_scaled=True)
               results_n_b = simu.simulate_auxiliary(x0=config_opc.PARA_X0, trajectory_ref=tra_ref, control_method="given",given_input=u_n_b)
               # Save
               x_n_b, y_n_b, z_n_b, u_n_b, j_f_n_b, aero_info_n_b = results_n_b
               np.savez(f'{pic_folder}\\data_nomorphing_both.npz', x_n_b=x_n_b, y_n_b=y_n_b, z_n_b=z_n_b, u_n_b=u_n_b, j_f_n_b=j_f_n_b,aero_info_n_b=aero_info_n_b)

          # Morphing, major cost + fuel cost + manu cost
          if from_file_m_b:
                # Load
               with np.load(f'{file_folder_m_b}\\data_morphing_both.npz') as data_morphing_both:
                    keys = ['x_m_b', 'y_m_b', 'z_m_b', 'u_m_b', 'j_f_m_b', 'aero_info_m_b']
                    results_m_b = [data_morphing_both[key] for key in keys]
          else:
               x_optimal, y_optimal, z_optimal, u_optimal, j_optimal = opt.generate_PS_solution_scaled(x0=config_opc.PARA_X0,trajectory_ref=tra_ref, morphing_disabled=None, fun_obj=opt.function_objective_both)
               t, x, y, z, u_m_b = opt.interpolate_optimal_trajectory(x_optimal, y_optimal, z_optimal, u_optimal, j_optimal,t_switch=tra_ref["t_switch"], from_scaled=True)
               results_m_b = simu.simulate_auxiliary(x0=config_opc.PARA_X0, trajectory_ref=tra_ref, control_method="given",given_input=u_m_b)
               # Save
               x_m_b, y_m_b, z_m_b, u_m_b, j_f_m_b, aero_info_m_b = results_m_b
               np.savez(f'{pic_folder}\\data_morphing_both.npz', x_m_b=x_m_b, y_m_b=y_m_b, z_m_b=z_m_b, u_m_b=u_m_b, j_f_m_b=j_f_m_b,aero_info_m_b=aero_info_m_b)
          
          pu.plot_comparison_open_morphing(pic_folder, 
                                        result_nomorphing=results_n, result_morphing=results_m,
                                        result_nomorphing_fuel=results_n_f, result_morphing_fuel=results_m_f,
                                        result_nomorphing_manu=results_n_m, result_morphing_manu=results_m_m,
                                        result_nomorphing_both=results_n_b, result_morphing_both=results_m_b,
                                        trajectory_ref=tra_ref, shown=shown)
     
     

     # from_file = False
     # cur_time = time.strftime("%Y%m%d-%H%M", time.localtime())
     # pic_folder = "pics\\results_open_"+cur_time
     # if not os.path.exists(pic_folder):
     #      os.mkdir(pic_folder)
     # if from_file:
     #      pu.plot_comparison_open_morphing(pic_folder=pic_folder, from_file=True, file_folder="pics\\results_open_20241215-1614")
     # else:
     #      # tra_ref = simu.generate_ref_trajectory_constant(constant_height=300)
     #      switch_time = 0.5
     #      constant_height=300
     #      tra_ref = simu.generate_ref_trajectory_varying(constant_height=constant_height, switch_time=switch_time,
     #                                                        high_height=350, low_height=250)

     #      # Non-Morphing, major cost
     #      x_optimal, y_optimal, z_optimal, u_optimal, j_optimal = opt.generate_PS_solution_casadi(x0=config_opc.PARA_X0,
     #                                                                                                trajectory_ref=tra_ref,
     #                                                                                                morphing_disabled=0.5)
     #      t, x, y, z, u_n = opt.interpolate_optimal_trajectory(x_optimal, y_optimal, z_optimal, u_optimal, j_optimal,
     #                                                             t_switch=tra_ref["t_switch"])
     #      results_n = simu.simulate_auxiliary(x0=config_opc.PARA_X0, trajectory_ref=tra_ref, control_method="given",
     #                                              given_input=u_n)

     #      # Morphing, major cost
     #      x_optimal, y_optimal, z_optimal, u_optimal, j_optimal = opt.generate_PS_solution_casadi(x0=config_opc.PARA_X0,
     #                                                                                                trajectory_ref=tra_ref,
     #                                                                                                morphing_disabled=None)
     #      t, x, y, z, u_m = opt.interpolate_optimal_trajectory(x_optimal, y_optimal, z_optimal, u_optimal, j_optimal,
     #                                                             t_switch=tra_ref["t_switch"])
     #      results_m = simu.simulate_auxiliary(x0=config_opc.PARA_X0, trajectory_ref=tra_ref, control_method="given",
     #                                              given_input=u_m)

     #      # Non-Morphing, major cost + fuel cost
     #      x_optimal, y_optimal, z_optimal, u_optimal, j_optimal = opt.generate_PS_solution_casadi(x0=config_opc.PARA_X0,
     #                                                                                                trajectory_ref=tra_ref,
     #                                                                                                morphing_disabled=0.5,
     #                                                                                                fun_obj=opt.function_objective_casadi_fuel)
     #      t, x, y, z, u_n_f = opt.interpolate_optimal_trajectory(x_optimal, y_optimal, z_optimal, u_optimal, j_optimal,
     #                                                             t_switch=tra_ref["t_switch"])
     #      results_n_f = simu.simulate_auxiliary(x0=config_opc.PARA_X0, trajectory_ref=tra_ref, control_method="given",
     #                                              given_input=u_n_f)

     #      # Morphing, major cost + fuel cost
     #      x_optimal, y_optimal, z_optimal, u_optimal, j_optimal = opt.generate_PS_solution_casadi(x0=config_opc.PARA_X0,
     #                                                                                                trajectory_ref=tra_ref,
     #                                                                                                morphing_disabled=None,
     #                                                                                                fun_obj=opt.function_objective_casadi_fuel)
     #      t, x, y, z, u_m_f = opt.interpolate_optimal_trajectory(x_optimal, y_optimal, z_optimal, u_optimal, j_optimal,
     #                                                             t_switch=tra_ref["t_switch"])
     #      results_m_f = simu.simulate_auxiliary(x0=config_opc.PARA_X0, trajectory_ref=tra_ref, control_method="given",
     #                                              given_input=u_m_f)

     #      # Non-Morphing, major cost + manu cost
     #      x_optimal, y_optimal, z_optimal, u_optimal, j_optimal = opt.generate_PS_solution_casadi(x0=config_opc.PARA_X0,
     #                                                                                                trajectory_ref=tra_ref,
     #                                                                                                morphing_disabled=0.5,
     #                                                                                                fun_obj=opt.function_objective_casadi_manu)
     #      t, x, y, z, u_n_m = opt.interpolate_optimal_trajectory(x_optimal, y_optimal, z_optimal, u_optimal, j_optimal,
     #                                                             t_switch=tra_ref["t_switch"])
     #      results_n_m = simu.simulate_auxiliary(x0=config_opc.PARA_X0, trajectory_ref=tra_ref, control_method="given",
     #                                              given_input=u_n_m)

     #      # Morphing, major cost + manu cost
     #      x_optimal, y_optimal, z_optimal, u_optimal, j_optimal = opt.generate_PS_solution_casadi(x0=config_opc.PARA_X0,
     #                                                                                                trajectory_ref=tra_ref,
     #                                                                                                morphing_disabled=None,
     #                                                                                                fun_obj=opt.function_objective_casadi_manu)
     #      t, x, y, z, u_m_m = opt.interpolate_optimal_trajectory(x_optimal, y_optimal, z_optimal, u_optimal, j_optimal,
     #                                                             t_switch=tra_ref["t_switch"])
     #      results_m_m = simu.simulate_auxiliary(x0=config_opc.PARA_X0, trajectory_ref=tra_ref, control_method="given",
     #                                              given_input=u_m_m)

     #      # Non-Morphing, major cost + fuel cost + manu cost
     #      x_optimal, y_optimal, z_optimal, u_optimal, j_optimal = opt.generate_PS_solution_casadi(x0=config_opc.PARA_X0,
     #                                                                                                trajectory_ref=tra_ref,
     #                                                                                                morphing_disabled=0.5,
     #                                                                                                fun_obj=opt.function_objective_casadi_both)
     #      t, x, y, z, u_n_b = opt.interpolate_optimal_trajectory(x_optimal, y_optimal, z_optimal, u_optimal, j_optimal,
     #                                                             t_switch=tra_ref["t_switch"])
     #      results_n_b = simu.simulate_auxiliary(x0=config_opc.PARA_X0, trajectory_ref=tra_ref, control_method="given",
     #                                              given_input=u_n_b)

     #      # Morphing, major cost + fuel cost + manu cost
     #      x_optimal, y_optimal, z_optimal, u_optimal, j_optimal = opt.generate_PS_solution_casadi(x0=config_opc.PARA_X0,
     #                                                                                                trajectory_ref=tra_ref,
     #                                                                                                morphing_disabled=None,
     #                                                                                                fun_obj=opt.function_objective_casadi_both)
     #      t, x, y, z, u_m_b = opt.interpolate_optimal_trajectory(x_optimal, y_optimal, z_optimal, u_optimal, j_optimal,
     #                                                             t_switch=tra_ref["t_switch"])
     #      results_m_b = simu.simulate_auxiliary(x0=config_opc.PARA_X0, trajectory_ref=tra_ref, control_method="given",
     #                                              given_input=u_m_b)

     #      pu.plot_comparison_open_morphing(pic_folder, 
     #                                         result_nomorphing=results_n, result_morphing=results_m,
     #                                         result_nomorphing_fuel=results_n_f, result_morphing_fuel=results_m_f,
     #                                         result_nomorphing_manu=results_n_m, result_morphing_manu=results_m_m,
     #                                         result_nomorphing_both=results_n_b, result_morphing_both=results_m_b,
     #                                         trajectory_ref=tra_ref)


