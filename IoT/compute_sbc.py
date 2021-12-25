from cmdstanpy import CmdStanModel
import numpy as np


def get_sbc_csv(stan_file, df, N):
    sbc_model = CmdStanModel(stan_file=stan_file)
    new_ranks = []
    for i in range(N):
        result_sbc = sbc_model.sample(
            data={"N_batch": 4, "N": 200, "batch": df.batch.values}
        )
        ranks = np.sum(
            result_sbc.stan_variable("lt_sim")[np.arange(0, 4000 - 7, 8)], axis=0
        )
        new_ranks.append(ranks)
    data = np.array(new_ranks)
    np.savetxt("new_ranks_N{}.csv".format(N), data.astype(int), delimiter=",", fmt="%i")
