from dvi_ekf import Config, Simulator

config = Config("config.yaml")
sim = Simulator(config)

if __name__ == "__main__":
    sim.run_once()
    sim.plot(compact=True)
