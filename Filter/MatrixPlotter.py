import matplotlib.pyplot as plt

class MatrixPlotter(object):
    def __init__(self, name, t0, mat):
        self.name = name
        self.mat = mat
        self.num_rows, self.num_cols = mat.shape
        self.t = [t0]

        # set initial values
        for i in range(self.num_rows):
            for j in range(self.num_cols):
                self.__dict__[f"a{i}{j}"] = [mat[i][j]]

    def __iter__(self):
        """ Make object iterable. """
        for attr, value in self.__dict__.items():
            yield attr, value

    def append(self, t, mat):
        self.t.append(t)

        for i in range(self.num_rows):
            for j in range(self.num_cols):
                self.__dict__[f"a{i}{j}"].append(mat[i][j])

    def plot(self, axes=None, min_t=None, max_t=None):
        """ Creates a plot of them matrix entries. """

        num_labels = self.num_rows * self.num_cols

        if axes is None:
            fig, axes = plt.subplots(self.num_rows, self.num_cols)
            fig.tight_layout()
            st = fig.suptitle(self.name, fontsize="x-large")

            # shift subplots down:
            st.set_y(0.95)
            fig.subplots_adjust(top=0.85)

        for row in range(self.num_rows):
            for col in range(self.num_cols):
                label = f"a{row}{col}"
                latex_label = f"$a_{{{row}{col}}}$"

                axes[row][col].plot(self.t, self.__dict__[label])

                axes[row][col].set_title(latex_label)
                axes[row][col].set_xlim(left=min_t, right=max_t)
                axes[row][col].grid(True)

        return axes