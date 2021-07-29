import matplotlib.pyplot as plt

class MatrixPlotter(object):
    def __init__(self, name,
            min_row = 0,
            min_col = 0,
            max_row = None,
            max_col = None):
        self.name = name
        self.t = []

        # min/max values
        self.min_row = min_row
        self.min_col = min_col
        self.max_row = max_row
        self.max_col = max_col

        self._init_matrix()

    def __iter__(self):
        """ Make object iterable. """
        for attr, value in self.__dict__.items():
            yield attr, value

    def _init_matrix(self):
        for i in range(self.min_row, self.max_row):
            for j in range(self.min_col, self.max_col):
                self.__dict__[f"a_{i}_{j}"] = []

    def append(self, t, mat):
        self.t.append(t)

        for i in range(self.min_row, self.max_row):
            for j in range(self.min_col, self.max_col):
                self.__dict__[f"a_{i}_{j}"].append(mat[i][j])

    def plot(self, axes=None, min_t=None, max_t=None, index_from_zero=False):
        """ Creates a plot of them matrix entries.

            Param:
            * index_from_zero - whether to label matrix entries starting from
                    0 or starting from 1
        """

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
                latex_label = f"$a_{{{row}\_{col}}}$" if index_from_zero \
                            else f"$a_{{{row+1}\_{col+1}}}$"

                val = self.__dict__[label]
                min_val, max_val = min(val), max(val)
                range_val = max_val - min_val

                # display of very small values
                if range_val < 0.0001:
                    min_val = min_val - 0.002
                    max_val = max_val + 0.002
                else:
                    min_val = min_val - 0.2 * range_val
                    max_val = max_val + 0.2 * range_val

                if plot_rows > 1:
                    ax_obj = axes[plt_row][plt_col]
                else:
                    ax_obj = axes[plt_col]

                ax_obj.plot(self.t, self.__dict__[label])

                ax_obj.set_title(latex_label)
                ax_obj.set_xlim(left=min_t, right=max_t)
                ax_obj.set_ylim(bottom=min_val, top=max_val)
                ax_obj.grid(True)

        return axes