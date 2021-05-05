import matplotlib.pyplot as plt

class MatrixPlotter(object):
    def __init__(self, name, t0, mat,
            min_row = 0,
            min_col = 0,
            max_row = None,
            max_col = None):
        self.name = name
        self.num_rows, self.num_cols = mat.shape
        self.t = [t0]

        # min/max values
        self.min_row = min_row
        self.min_col = min_col
        self.max_row = (self.num_rows if max_row is None else max_row)
        self.max_col = (self.num_cols if max_col is None else max_col)

        # set initial values
        for i in range(self.min_row, self.max_row):
            for j in range(self.min_col, self.max_col):
                self.__dict__[f"a_{i}_{j}"] = [mat[i][j]]

    def __iter__(self):
        """ Make object iterable. """
        for attr, value in self.__dict__.items():
            yield attr, value

    def append(self, t, mat):
        self.t.append(t)

        for i in range(self.min_row, self.max_row):
            for j in range(self.min_col, self.max_col):
                self.__dict__[f"a_{i}_{j}"].append(mat[i][j])

    def plot(self, axes=None, min_t=None, max_t=None):
        """ Creates a plot of them matrix entries. """

        plot_rows = self.max_row - self.min_row
        plot_cols = self.max_col - self.min_col

        if axes is None:
            fig, axes = plt.subplots(plot_rows, plot_cols)
            fig.tight_layout()
            st = fig.suptitle(self.name, fontsize="x-large")

            # shift subplots down:
            st.set_y(0.95)
            fig.subplots_adjust(top=0.85)

        for row in range(self.min_row, self.max_row):
            plt_row = row - self.min_row

            for col in range(self.min_col, self.max_col):
                plt_col = col - self.min_col

                label = f"a_{row}_{col}"
                latex_label = f"$a_{{{row}\_{col}}}$"

                axes[plt_row][plt_col].plot(self.t, self.__dict__[label])

                axes[plt_row][plt_col].set_title(latex_label)
                axes[plt_row][plt_col].set_xlim(left=min_t, right=max_t)
                axes[plt_row][plt_col].grid(True)

        return axes