import matplotlib.pyplot as plt
import math

def show_plot(plot_func, *args):
    def wrapper(*args):
        plot_func(*args)
        plt.show()
    return wrapper

def plot_loop(f_plot_objs, **kwargs):
    def make_vals_dict(objs, vals):
        def make_val_dict(obj, val):
            if obj and (val is not []):
                return {'t' : obj.t, 'vals' : val,}

        return {o.name: make_val_dict(o, vals[i])
                for i, o in enumerate(objs) if o is not None}

    def get_flat_vals(vals_dict, vals):
        if len(vals_dict.keys()) == 1:
            return vals
        else:
            return [v for sv in vals for v in sv]

    def wrapper(self, **kwargs):
        labels      = kwargs['labels']
        num_cols    = kwargs['num_cols']
        axes        = kwargs['axes']
        filename    = kwargs['filename']
        config      = kwargs['config']
        
        Q = config.meas_noise

        num_rows    = math.ceil( len(labels) / num_cols )
        axes        = self._init_axes(axes, num_rows, num_cols)

        ai = 0
        for i, label in enumerate(labels):
            # plot settings
            if label in self.jump_labels:
                ai += 1

            if i == 0:
                offset = ai

            a = self._get_ax(axes, ai, num_rows)

            # pre-processing of data
            objs, vals = f_plot_objs(self, label, **kwargs)
            vals_dict = make_vals_dict(objs, vals)
            flat_vals = get_flat_vals(vals_dict, vals)

            # actual plotting
            self._plot_vals(a, vals_dict)

            # post-processing the plot
            self._ax_postfix(a, label, Q, *flat_vals)
            ai += 1

        self._fig_postfix(filename, axes, offset, num_rows, config)
        return axes

    return wrapper