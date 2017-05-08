#!/usr/bin/env python

import matplotlib
matplotlib.use('pgf')

import numpy as np
import matplotlib.pyplot as plt
import csv
from scipy import stats
import math
from matplotlib2tikz import save as tikz_save
import argparse
import os

version = "0.3"

# IO ###############################################################


def read_csv(name, indices):
    """
    Parameters
    ----------
    name: string
        a path to a CSV file
    indices: array_like
        a list of ints corresponding to the column no. of the data
    Returns
    -------
    data: ndarray
        an array of the selected columns according to indices
    header: a list of headers from the first row of CSV, and columns according to indices
    """
    with open(name) as csvfile:
        table = list(csv.reader(csvfile))
    # get labels
    header = table.pop(0)
    header = [header[i] for i in indices]
    # get x, y
    data = np.array([[float(item[i]) for item in table] for i in indices])
    return (data, header)


def stat2tsv(statname, data_stat):
    """
    Parameters
    ----------
    statname: string
        a path to the output filename
    data_stat: ndarray
        a 2D array where each row has A, slope, intercept, r_square, r_value, p_value, std_err
        corresponds to the stat of each plot
    """
    np.savetxt(statname, data_stat, delimiter='\t', newline='\n',
               header='A\tslope\tintercept\tr_square\tr_value\tp_value\tstd_err')

# tools ##################################################################


def get_stat(data, scaling):
    """
    Parameters
    ----------
    data: ndarray
        first row corresponds to x-axis
        other rows are different data for the y-values
    scaling: string
        'loglog' or 'plot', corresponds to log-log scale or linear scale
    Returns
    -------
    data_fit: ndarray
        The linear regression fit of each y-value rows from data
        i.e. it has 1 lesser row than data
    data_stat: ndarray
        a 2D array where each row has A, slope, intercept, r_square, r_value, p_value, std_err
        corresponds to the stat of each plot
    """
    # get natural logs
    if scaling == "loglog":
        data_plot = np.log(data)
    elif scaling == "plot":
        data_plot = data
    # Linear regression
    # m is the no. of plots
    m = data.shape[0]
    # n is the no. of data per plots
    n = data.shape[1]
    data_stat = np.empty((m - 1, 7))
    data_fit = np.empty((m - 1, n))
    for i in range(1, m):
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            data_plot[0], data_plot[i])
        # get the fit as a function of x
        if scaling == "loglog":
            # get prefactor: log y = m log x + c. y = A x^m, A = e^c.
            A = math.exp(intercept)
            fit = A * data[0] ** slope
        elif scaling == "plot":
            # create A for output only
            A = None
            fit = intercept + slope * data[0]
        r_square = r_value ** 2
        data_stat[i - 1, :] = np.array([A, slope, intercept,
                                        r_square, r_value, p_value, std_err])
        data_fit[i - 1, :] = fit[:]
    return (data_fit, data_stat)

# matplotlib pgf backend


def plot(outname, outputExt, header, data, data_fit, scaling):
    """
    Parameters
    ----------
    outname: string
        output file name
    outputExt: string
        output file extension, e.g. pdf
    header: list of string
        the labels of individual data rows
    data: ndarray
        first row corresponds to x-axis
        other rows are different data for the y-values
    data_fit: ndarray
        The linear regression fit of each y-value rows from data
        i.e. it has 1 lesser row than data
    scaling: string
        'loglog' or 'plot', corresponds to log-log scale or linear scale
    Returns
    -------
    None: but this function will write to a file named outname
    """
    if outputExt == "pgf" or outputExt == "pdf":
        # setup fonts using Latin Modern
        pgf_with_rc_fonts = {
            "font.family": ["serif"],
            "font.serif": ["Latin Modern Roman"],
            "font.sans-serif": ["Latin Modern Sans"],
            "font.monospace": ["Latin Modern Mono"]
        }
        matplotlib.rcParams.update(pgf_with_rc_fonts)
    n = len(header)

    color = iter(plt.cm.rainbow(np.linspace(0, 1, n)))
    # plot
    for i in range(1, n):
        markerfacecolor = next(color)
        if scaling == "loglog":
            plt.loglog(data[0], data[i], marker="o",
                       markerfacecolor=markerfacecolor, linestyle="None", label=header[i])
            plt.loglog(data[0], data_fit[i - 1], '--k')
        elif scaling == "plot":
            plt.plot(data[0], data[i], marker="o",
                     markerfacecolor=markerfacecolor, linestyle="None", label=header[i])
            plt.plot(data[0], data_fit[i - 1], '--k')
    plt.xlabel(header[0])

    # legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    # plt.text(50., 7, "R = {}".format(r_value), family="serif") # put stat in
    # the plot?
    if outputExt == "tikz":
        tikz_save(outname,
                  figureheight='\\figureheight',
                  figurewidth='\\figurewidth')
    else:
        plt.savefig(outname)

########################################################################


def main(args):
    # get args
    name = args.f
    outputStat = args.s
    scaling = args.S
    # x, y indices: column no. for x and y values
    # y can have multiple values
    indices = [int(y) for y in args.y.split(',')]
    indices.insert(0, int(args.x))
    # extensions
    if args.o:
        outputExt = os.path.splitext(args.o)[1][1:]
    else:
        outputExt = args.t
    # output filenames
    if args.o:
        nameWOExt = os.path.splitext(args.o)[0]
    else:
        nameWOExt = os.path.splitext(name)[0]
    outname = ".".join((nameWOExt, outputExt))

    # plot
    data, header = read_csv(name, indices)

    data_fit, data_stat = get_stat(data, scaling)

    # plot to outname
    plot(outname, outputExt, header, data, data_fit, scaling)

    # output stat to txt
    if outputStat:
        statname = ".".join((nameWOExt, "tsv"))
        stat2tsv(statname, data_stat)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.set_defaults(func=main)
    # define args
    parser.add_argument('-v', action='version', version=version)
    parser.add_argument('-f', metavar="input filename", help='CSV input.')
    parser.add_argument('-t', metavar="extension",
                        help='Output format: tikz/pgf/pdf.', default="pdf")
    parser.add_argument('-o', metavar="output filename",
                        help='Output filename (Optional).')
    parser.add_argument('-S', metavar="scaling",
                        help='Scaling of the plot: plot/loglog.', default="loglog")
    parser.add_argument('-x', metavar="x-column",
                        help='Column no. for x (horizontal) value.', default=0)
    parser.add_argument('-y', metavar="y-column",
                        help='Column no. for y (vertical) value.', default=1)
    parser.add_argument('-s', action='store_true',
                        help='If specified, output statistics to .txt.')
    # parsing and run main
    args = parser.parse_args()
    args.func(args)
