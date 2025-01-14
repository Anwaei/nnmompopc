import numpy as np
import dynamics as dyn
import config_opc
import plot_utils as pu
import simulate as simu
import optimal as opt
from matplotlib import pyplot as plt
import time
import os

if __name__ == '__main__':
     from_file = False
     cur_time = time.strftime("%Y%m%d-%H%M", time.localtime())
     pic_folder = "pics\\results_open_"+cur_time
     if not os.path.exists(pic_folder):
          os.mkdir(pic_folder)
     if from_file:
          pu.plot_comparison_open_morphing(pic_folder=pic_folder, from_file=True, file_folder="pics\\results_open_20241215-1614")
     else:
          # tra_ref = simu.generate_ref_trajectory_constant(constant_height=300)
          switch_time = 0.5
          constant_height=300
          tra_ref = simu.generate_ref_trajectory_varying(constant_height=constant_height, switch_time=switch_time,
                                                            high_height=350, low_height=250)

          # Non-Morphing, major cost
          x_optimal, y_optimal, z_optimal, u_optimal, j_optimal = opt.generate_PS_solution_casadi(x0=config_opc.PARA_X0,
                                                                                                    trajectory_ref=tra_ref,
                                                                                                    morphing_disabled=0.5)
          t, x, y, z, u_n = opt.interpolate_optimal_trajectory(x_optimal, y_optimal, z_optimal, u_optimal, j_optimal,
                                                                 t_switch=tra_ref["t_switch"])
          results_n = simu.simulate_auxiliary(x0=config_opc.PARA_X0, trajectory_ref=tra_ref, control_method="given",
                                                  given_input=u_n)

          # Morphing, major cost
          x_optimal, y_optimal, z_optimal, u_optimal, j_optimal = opt.generate_PS_solution_casadi(x0=config_opc.PARA_X0,
                                                                                                    trajectory_ref=tra_ref,
                                                                                                    morphing_disabled=None)
          t, x, y, z, u_m = opt.interpolate_optimal_trajectory(x_optimal, y_optimal, z_optimal, u_optimal, j_optimal,
                                                                 t_switch=tra_ref["t_switch"])
          results_m = simu.simulate_auxiliary(x0=config_opc.PARA_X0, trajectory_ref=tra_ref, control_method="given",
                                                  given_input=u_m)

          # Non-Morphing, major cost + fuel cost
          x_optimal, y_optimal, z_optimal, u_optimal, j_optimal = opt.generate_PS_solution_casadi(x0=config_opc.PARA_X0,
                                                                                                    trajectory_ref=tra_ref,
                                                                                                    morphing_disabled=0.5,
                                                                                                    fun_obj=opt.function_objective_casadi_fuel)
          t, x, y, z, u_n_f = opt.interpolate_optimal_trajectory(x_optimal, y_optimal, z_optimal, u_optimal, j_optimal,
                                                                 t_switch=tra_ref["t_switch"])
          results_n_f = simu.simulate_auxiliary(x0=config_opc.PARA_X0, trajectory_ref=tra_ref, control_method="given",
                                                  given_input=u_n_f)

          # Morphing, major cost + fuel cost
          x_optimal, y_optimal, z_optimal, u_optimal, j_optimal = opt.generate_PS_solution_casadi(x0=config_opc.PARA_X0,
                                                                                                    trajectory_ref=tra_ref,
                                                                                                    morphing_disabled=None,
                                                                                                    fun_obj=opt.function_objective_casadi_fuel)
          t, x, y, z, u_m_f = opt.interpolate_optimal_trajectory(x_optimal, y_optimal, z_optimal, u_optimal, j_optimal,
                                                                 t_switch=tra_ref["t_switch"])
          results_m_f = simu.simulate_auxiliary(x0=config_opc.PARA_X0, trajectory_ref=tra_ref, control_method="given",
                                                  given_input=u_m_f)

          # Non-Morphing, major cost + manu cost
          x_optimal, y_optimal, z_optimal, u_optimal, j_optimal = opt.generate_PS_solution_casadi(x0=config_opc.PARA_X0,
                                                                                                    trajectory_ref=tra_ref,
                                                                                                    morphing_disabled=0.5,
                                                                                                    fun_obj=opt.function_objective_casadi_manu)
          t, x, y, z, u_n_m = opt.interpolate_optimal_trajectory(x_optimal, y_optimal, z_optimal, u_optimal, j_optimal,
                                                                 t_switch=tra_ref["t_switch"])
          results_n_m = simu.simulate_auxiliary(x0=config_opc.PARA_X0, trajectory_ref=tra_ref, control_method="given",
                                                  given_input=u_n_m)

          # Morphing, major cost + manu cost
          x_optimal, y_optimal, z_optimal, u_optimal, j_optimal = opt.generate_PS_solution_casadi(x0=config_opc.PARA_X0,
                                                                                                    trajectory_ref=tra_ref,
                                                                                                    morphing_disabled=None,
                                                                                                    fun_obj=opt.function_objective_casadi_manu)
          t, x, y, z, u_m_m = opt.interpolate_optimal_trajectory(x_optimal, y_optimal, z_optimal, u_optimal, j_optimal,
                                                                 t_switch=tra_ref["t_switch"])
          results_m_m = simu.simulate_auxiliary(x0=config_opc.PARA_X0, trajectory_ref=tra_ref, control_method="given",
                                                  given_input=u_m_m)

          # Non-Morphing, major cost + fuel cost + manu cost
          x_optimal, y_optimal, z_optimal, u_optimal, j_optimal = opt.generate_PS_solution_casadi(x0=config_opc.PARA_X0,
                                                                                                    trajectory_ref=tra_ref,
                                                                                                    morphing_disabled=0.5,
                                                                                                    fun_obj=opt.function_objective_casadi_both)
          t, x, y, z, u_n_b = opt.interpolate_optimal_trajectory(x_optimal, y_optimal, z_optimal, u_optimal, j_optimal,
                                                                 t_switch=tra_ref["t_switch"])
          results_n_b = simu.simulate_auxiliary(x0=config_opc.PARA_X0, trajectory_ref=tra_ref, control_method="given",
                                                  given_input=u_n_b)

          # Morphing, major cost + fuel cost + manu cost
          x_optimal, y_optimal, z_optimal, u_optimal, j_optimal = opt.generate_PS_solution_casadi(x0=config_opc.PARA_X0,
                                                                                                    trajectory_ref=tra_ref,
                                                                                                    morphing_disabled=None,
                                                                                                    fun_obj=opt.function_objective_casadi_both)
          t, x, y, z, u_m_b = opt.interpolate_optimal_trajectory(x_optimal, y_optimal, z_optimal, u_optimal, j_optimal,
                                                                 t_switch=tra_ref["t_switch"])
          results_m_b = simu.simulate_auxiliary(x0=config_opc.PARA_X0, trajectory_ref=tra_ref, control_method="given",
                                                  given_input=u_m_b)

          pu.plot_comparison_open_morphing(pic_folder, 
                                             result_nomorphing=results_n, result_morphing=results_m,
                                             result_nomorphing_fuel=results_n_f, result_morphing_fuel=results_m_f,
                                             result_nomorphing_manu=results_n_m, result_morphing_manu=results_m_m,
                                             result_nomorphing_both=results_n_b, result_morphing_both=results_m_b,
                                             trajectory_ref=tra_ref)
