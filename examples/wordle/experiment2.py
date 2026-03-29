from tenyson import run_experiment


def build(exp):
    seed = exp.seed("experiment2_sft")
    exp.eval("baseline", run="eval_turn6_prime", adapter=seed)
    exp.rl("train", run="rl_turn6_prime", adapter=seed)
    exp.eval("final", run="eval_turn6_prime", adapter=exp.adapter("train"))


if __name__ == "__main__":
    run_experiment(__file__, build)
