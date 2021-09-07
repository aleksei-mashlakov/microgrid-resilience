from multiprocessing import Pool
import cvxpy as cvx
from cvxstoc import NormalRandomVariable, prob, expectation
from cvxpower import FixedLoad, Generator, Net, Group, Device, Terminal
from datetime import timedelta
import pandas as pd
import numpy as np
import dccp
import time
import os

class LossyStorage(Device):
    r"""Storage device.
    A storage device either takes or delivers power with charging and
    discharging rates specified by the constraints
    .. math::
      -D^\max \le p \le C^\max
    ..
    where :math:`C^\max` and :math:`D^\max` are the maximum charging and
    discharging rates. The charge level of the battery is given by
    .. math::
      q(\tau) = q^\mathrm{init} +  \sum_{t=1}^\tau p(t), \quad \tau = 1, \ldots, T,
    ..
    which is constrained according to the physical limits of the battery
    .. math::
      0 \le q \le Q^\max.
    ..
    :param discharge_max: Maximum discharge rate, :math:`D^\max`
    :param charge_max: Maximum charge rate, :math:`C^\max`
    :param energy_init: Initial charge, :math:`q^\mathrm{init}`
    :param energy_max: Maximum battery capacity, :math:`Q^\max`
    :param name: (optional) Display name of storage device
    :type discharge_max: float or sequence of floats
    :type charge_max: float or sequence of floats
    :type energy_init: float
    :type energy_max: float or sequence of floats
    :type name: string
    """

    def __init__(
        self,
        discharge_max=0,
        charge_max=None,
        energy_init=0,
        energy_final=None,
        energy_max=None,
        name=None,
        len_interval=1.0,
        final_energy_price=None,
        alpha = 0.0,
        DoD = 0.9
    ):
        super(LossyStorage, self).__init__([Terminal()], name)
        self.discharge_max = discharge_max
        self.charge_max = charge_max
        self.energy_init = energy_init
        self.energy_max = energy_max
        self.energy_min = energy_max * (1 - DoD)
        self.energy_final = energy_final
        self.len_interval = len_interval  # in hours
        self.final_energy_price = final_energy_price
        self.energy = None
        self.alpha = alpha
        self.T = int(self.len_interval*24)

    @property
    def cost(self):
        T, S = self.terminals[0].power_var.shape
        if self.final_energy_price is not None:
            if self.energy is None:
                self.energy = cvx.Variable(self.terminals[0].power_var.shape)
            cost = np.zeros((T - 1, S))
            final_cost = cvx.reshape(
                self.energy[-1, :] * self.final_energy_price[0, 0], (1, S)
            )
            cost = cvx.vstack([cost, final_cost])
        else:
            cost = np.zeros(T, S)
        return cost

    @property
    def constraints(self):
        P = self.terminals[0].power_var
        if self.energy is None:
            self.energy = cvx.Variable(self.terminals[0].power_var.shape)
        e_init = cvx.reshape(self.energy_init, ())
        constr = [
            #cvx.diff(self.energy.T) == P[1:, :] * self.len_interval,
            #self.energy[0, :] - e_init - P[0, :] * self.len_interval == 0,
            self.energy[0, :] == e_init,
            self.terminals[0].power_var >= -self.discharge_max,
            self.terminals[0].power_var <= self.charge_max,
            self.energy <= self.energy_max,
            self.energy >= self.energy_min,
        ]
        for t in range(0, self.T - 1):
            constr += [
                self.energy[t+1] == (1 - self.alpha) * self.energy[t] + P[t] * self.len_interval
            ]
        if self.energy_final is not None:
            constr += [(1 - self.alpha) * self.energy[-1] + P[-1] * self.len_interval>= self.energy_final]
            #constr += [self.energy[-1] >= self.energy_final]
        return constr


class Converter(Device):
    """Storage converter.
    A loss converter has two terminals with power schedules
    :math:`p_1` and :math:`p_2`. Conservation of energy across the
    converter is enforced with the constraint
    .. math::
      p_1 + p_2 = (1-eta)p_1,
      p_1 + p_2 = (1-eta)p_2,
    ..
    and a maximum capacity of :math:`P^\max` with
    .. math::
      |p_1| \le P^\max.
    ..
    :param power_max: Maximum capacity of the converter line
    :param name: (optional) Display name for converter line
    :type power_max: float or sequence of floats
    :type name: string
    """

    def __init__(self, eta=0.0, power_max=None, name=None):
        super(Converter, self).__init__([Terminal(), Terminal()], name)
        self.power_max = power_max
        self.eta = eta
        assert self.eta >= 0

    @property
    def constraints(self):
        p1 = self.terminals[0].power_var
        p2 = self.terminals[1].power_var

        constrs = []
        if self.eta > 0:
            constrs += [p1 + p2 >= (1 - self.eta)*p1]
#             constrs += [p1 + p2 <= (1 - self.eta)*p2]
            if self.power_max is not None:
                constrs += [((2 * self.power_max) * (1 - self.eta)) / (1 + self.eta) >= p1 + p2]
#                 constrs += [((2 * self.power_max) * (self.eta - 1)) / (1 + self.eta) >= p1 + p2]
        else:
            constrs += [p1 + p2 == 0]
            if self.power_max is not None:
                constrs += [cvx.abs((p1 - p2) / 2) <= self.power_max]

        return constrs

def getConstraints(network):
    constraints = []
    group = network.devices + network.nets
    [constraints.append(constraint) for x in group for constraint in x.constraints]
    return constraints

def getData(df, day, d_name):
    idx = df[df.index.dayofyear==day].index
    return df.loc[idx, d_name].values

def process_image(NUM_HOURS_ENS):

    cov_matrix_load = pd.read_csv('./data/cov_matrix_load.csv').values
    #cov_matrix_load = np.repeat(np.repeat(pd.read_csv('./cov_matrix_load.csv').values, 2, axis=0), 2, axis=1)
    #cov_matrix_solar = np.repeat(np.repeat(pd.read_csv('./cov_matrix_solar.csv').values, 2, axis=0), 2, axis=1)
    cov_matrix_solar = pd.read_csv('./data/cov_matrix_solar.csv').values
    df_optim = pd.read_csv('./data/lvdc_microgrid_optim_variables_short.csv',
                           index_col=0, infer_datetime_format=True, parse_dates=["index"])
    # df_optim = df_optim.append(pd.DataFrame(index=[df_optim.index[-1]+timedelta(hours=1)]))
    # df_optim = df_optim.resample("30T").ffill().dropna() # resample to 30 minutes
    df_acc = pd.read_csv('./data/accuracy_in_clusters_daily_predictions_2013.txt', header=None).rename(columns={0:'accuracy'})
    n_clusters = 5
    year = 2013
    n_largest = list(df_acc.sort_values(by='accuracy')[-n_clusters:].index)
    n_smalles = list(df_acc.sort_values(by='accuracy')[:n_clusters].index)

    df_lambda = pd.read_excel('./data/predictions_in_clusters_daily_{}.xlsx'.format(year))
    df_lambda.index = pd.to_datetime(df_lambda['date'] , format="%Y-%m-%d")
    df_lambda.drop(['date'], axis=1, inplace=True)
    INTERVAL_LENGTH = 1. # hours (1/6 hour = 10 minutes)
    T = int(24/INTERVAL_LENGTH)
    RATED_P = 65 # kW
    CHARGE_DISCHARGE_EFFICIENCY = 0.96 # 92 % round-trip efficiency
    SELF_DISCHARGE_EFFICIENCY = 0.001/T  # 0.10 % per day
    NUM_SUMPLES = 100
    TRAFO_LIMIT = 250
    ETA = 0.95
    C_rm = 0.0024
    C_nsc = 0.0521
    C_etax = 0.0279
    C_fic = 0.0007
    C_dns = 12.1521
    C_betta = 0.021 #0.042 #0.0817
    save_folder = "./results"
    Size_kWh = NUM_HOURS_ENS * RATED_P
    print("BESS size: {}".format(Size_kWh))
    BATTERY_ENERGY_MAX = Size_kWh # kWh
    #NUM_HOURS_ENS = int(BATTERY_ENERGY_MAX/ RATED_P)
    INITIAL_BATTERY_CHARGE = 0.5 * BATTERY_ENERGY_MAX  # kWh

    for cluster in n_smalles + n_largest:
        print('Cluster {}'.format(cluster))
        # save energy schedule
        df_energy_results = pd.DataFrame(columns = ['economic', 'reliable', 'predicted', 'perfect_foresight'],
                                  index = df_optim.index).add_suffix('_{}'.format(cluster))
        # save power schedule, p_net, and p_pcc
        df_power_results = pd.DataFrame(columns = ['economic', 'reliable', 'predicted', 'perfect_foresight'],
                                        index = df_optim.index)
        dfs_list = []
        for cols in ['p_bess', 'p_net', 'p_pcc']:
            dfs_list.append(df_power_results.add_suffix('_{}_{}'.format(cols,cluster)))
        df_power_results = pd.concat(dfs_list, axis=1)

        for day in list(df_optim.index.dayofyear.unique())[0]:
            print('Simulation day {}'.format(day))
            # get variables
            load_power = getData(df_optim, day, 'pred_load')
            load_variable = NormalRandomVariable(load_power, cov_matrix_load)
            solar_power = getData(df_optim, day, 'pred_solar')
            solar_variable = NormalRandomVariable(solar_power, cov_matrix_solar)

            # create devices
            load = FixedLoad(power=load_power.reshape(T,1), name='Load demand')
            solar = Generator(power_max=solar_power.reshape(T,1),
                              power_min=solar_power.reshape(T,1),
                              len_interval=INTERVAL_LENGTH, name="Solar PV")
            grid = Generator(power_max=TRAFO_LIMIT, power_min=-TRAFO_LIMIT,
                             len_interval=INTERVAL_LENGTH, name="Grid") #alpha=alpha, beta=beta, gamma=gamma,
            converter = Converter(power_max=RATED_P, eta=CHARGE_DISCHARGE_EFFICIENCY, name='Storage converter')
            storage = LossyStorage(discharge_max=RATED_P,
                                   charge_max=RATED_P,
                                   energy_max=BATTERY_ENERGY_MAX,
                                   energy_final=INITIAL_BATTERY_CHARGE,
                                   energy_init=INITIAL_BATTERY_CHARGE,
                                   len_interval=INTERVAL_LENGTH,
                                   alpha=SELF_DISCHARGE_EFFICIENCY,
                                   name='Storage')
            # create network
            net1 = Net([grid.terminals[0], converter.terminals[0], solar.terminals[0], load.terminals[0]], name = 'Bus 1')
            net2 = Net([converter.terminals[1], storage.terminals[0]], name = 'Bus 2')

            network = Group([grid, converter, storage, solar, load], [net1, net2])
            network.init_problem(time_horizon=T, num_scenarios=1)

            constraints = getConstraints(network)
            constrains_economic = getConstraints(network)


            sum_discharge = sum([SELF_DISCHARGE_EFFICIENCY**(x) for x in range(int(NUM_HOURS_ENS/INTERVAL_LENGTH))])
            self_discharge = cvx.Parameter(nonneg=True, value=sum_discharge)
            p_bess = (storage.energy[:,0]*(CHARGE_DISCHARGE_EFFICIENCY)/(self_discharge))/NUM_HOURS_ENS

            p_net = load_variable - solar_variable  - p_bess
            constraints += [prob(p_net >= 0, NUM_SUMPLES) <= 1 - ETA]

            C_export = getData(df_optim, day, 'market_price')/1.24 # excluding tax
            C_import = getData(df_optim, day, 'market_price') + C_rm + C_nsc + C_etax

            #print('Import price: {}'.format(C_import))
            #print('Export price: {}'.format(C_export))

            p_pcc = -grid.terminals[0].power_var
            C_pcc = (C_export @ (p_pcc) + (C_import-C_export) @ (p_pcc+cvx.abs(p_pcc))/2) * INTERVAL_LENGTH
            C_bess = np.repeat(C_betta,T) @ cvx.abs(storage.terminals[0].power_var) * INTERVAL_LENGTH

            f_e = C_pcc + C_bess
            f_r = np.repeat(C_dns,T) @ cvx.pos(p_net) * INTERVAL_LENGTH #/ T

            obj1_list = []
            obj2_list = []
            for scenario in ['economic', 'reliable', 'predicted']:
                print(scenario)
                if scenario == 'predicted':
                    idx = df_lambda[df_lambda.index.dayofyear==day].index
                    lmbda = df_lambda.loc[idx, cluster].values[0]
                elif scenario == 'economic':
                    lmbda = 0.0
                elif scenario == 'reliable':
                    lmbda = 1.0
                else:
                    pass

                #print('Trade-off coefficien: {}'.format(lmbda))

                if scenario != 'predicted':
                    f_r_expect = expectation(f_r, num_samples=NUM_SUMPLES)
                    f_obj = (1 - lmbda) * f_e + (lmbda) * f_r_expect
                    objective = cvx.Minimize(f_obj)
                    if scenario == 'economic':
                        problem = cvx.Problem(objective, constrains_economic)
                    else:
                        problem = cvx.Problem(objective, constraints)
                    try:
                        problem.solve(solver="ECOS")
                    except (cvx.error.SolverError,cvx.error.ParameterError) as e:
                        print("Exception: {}".format(e))
                        print(f"Problem status: {problem.status}")
                        print(f"Problem value: {problem.value}")
                    if problem.status in ["infeasible", "unbounded"]:
                        print(f"Problem status: {problem.status}")
                        break
                    else:
                        # print('f_e {}'.format(f_e.value[0]))
                        # print('f_r {}'.format(f_r_expect.value))
                        obj1_list.append(f_e.value[0])
                        obj2_list.append(f_r_expect.value)

                else:
                    print(obj1_list, obj2_list)
                    f_r_expect = expectation(f_r, num_samples=NUM_SUMPLES)
                    denom = np.max(obj1_list) - np.min(obj1_list)
                    if denom == 0.0:
                        f_e_norm = f_e
                    else:
                        f_e_norm = (f_e - np.min(obj1_list))/denom
                    denom = np.max(obj2_list) - np.min(obj2_list)
                    if denom == 0.0:
                        f_r_norm = f_r_expect
                    else:
                        f_r_norm = (f_r_expect - np.min(obj2_list))/denom

                    f_obj = (1 - lmbda) * f_e_norm + (lmbda) * f_r_norm
                    objective = cvx.Minimize(f_obj)
                    problem = cvx.Problem(objective, constraints)
                    try:
                        problem.solve(solver="ECOS")
                    except (cvx.error.SolverError,cvx.error.ParameterError) as e:
                        print("Exception: {}".format(e))
                    print(f"Problem status: {problem.status}")
                    print(f"Problem value: {problem.value}")
                    if problem.status in ["infeasible", "unbounded"]:
                        print(f"Problem status: {problem.status}")
                        break
                    else:
                        pass
                        # if not variable is None:
                        #     print('f_e {}'.format(f_e_norm.value[0]))
                        #     print('f_r {}'.format(f_r_norm.value))

                # save variables
                idx = df_energy_results[df_energy_results.index.dayofyear==day].index
                df_energy_results.loc[idx, scenario + '_{}'.format(cluster)] = storage.energy.value.reshape(-1)
                df_power_results.loc[idx, scenario + '_{}_{}'.format('p_net',cluster)] = expectation(p_net, num_samples=NUM_SUMPLES).value
                df_power_results.loc[idx, scenario + '_{}_{}'.format('p_pcc',cluster)] = -grid.terminals[0].power_var.value.reshape(-1)
                df_power_results.loc[idx, scenario + '_{}_{}'.format('p_bess',cluster)] = converter.terminals[0].power_var.value.reshape(-1)

        if not os.path.exists(f'{save_folder}'):
            os.makedirs(f'{save_folder}')
        df_power_results.reset_index().to_csv('{}/power_results_{}_{}.csv'.format(save_folder,cluster,NUM_HOURS_ENS), index=False)
        df_energy_results.reset_index().to_csv('{}/energy_results_{}_{}.csv'.format(save_folder,cluster,NUM_HOURS_ENS), index=False)
    return

#                 if not problem.status in ["infeasible", "unbounded", "infeasible_inaccurate", None]:
#                     #print('PCC power: {}'.format(p_pcc.value))
#                     #print('Storage charge: {}'.format(storage.energy.value))
#                     import matplotlib.pyplot as plt
#                     fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,3))
#                     ax.plot(storage.energy.value, '.-', label='storage energy')
#                     ax.plot(grid.terminals[0].power_var.value, color='blue', label='grid')
#                     ax.plot(solar.terminals[0].power_var.value, color='pink', label='solar')
#                     ax.plot(converter.terminals[0].power_var.value, color='red', label='battery')
#                     #ax.plot(p_net.value, color='yellow', label='p_net')
#                     ax.plot(p_pcc.value, color='yellow', label='p_pcc')
#                     ax.plot(load.terminals[0].power_var.value, color='green', label='load')
#                     ax.plot(C_import*500, label='price')
#                     ax.set_ylabel("stored energy")
#                     if scenario == 'predicted':
#                         ax.set_title("f_e {} f_r {} lmbda {} ".format(f_e_norm.value[0], f_r_norm.value, lmbda))
#                     else:
#                         ax.set_title("f_e {} f_r {} lmbda {} ".format(f_e.value[0], f_r_expect.value, lmbda))
#                     ax.legend()
#                     fig.show()

def main():
    #data_inputs = [100, 150, 200, 250, 300, 350, 400, 450, 500]
    data_inputs = list(range(1,4)) #list(np.arange(0.5, 3.5, 0.5)) # hours
    parallel_processes = 3
    pool = Pool(parallel_processes)       # Create a multiprocessing Pool
    pool.map(process_image, data_inputs)  # process data_inputs iterable with pool


if __name__ == '__main__':
    ## This part is the first to execute when script is ran. It times the execution time and rans the main function
    start = time.time()
    main()
    end = time.time()
    print("Script completed in " + str(end - start) + " seconds")
