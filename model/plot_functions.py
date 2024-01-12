import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from IPython.display import SVG, display
from rdkit import Chem
from rdkit.Chem import AllChem, rdmolops, rdCoordGen
from rdkit.Chem.Draw import rdMolDraw2D
from matplotlib import cm


# style
matplotlib.style.use("seaborn-v0_8-whitegrid")
matplotlib.style.use("seaborn-v0_8-pastel")

# font "Arial"
FONTFAMILY='Times New Roman'
sns.set(font_scale=1.4)
# color palette
# tab10, tab20(2 color each), tab20c(4 color each)
# Set2
# sns.set_palette('tab10')
sns.set_style('whitegrid')


########################################################################################################

COLOR_DICT = {
    'permeability': ['#fce85d', '#66c18c'],
    'testset': ['#7CB4EC', '#AFE9FF', '#9FE5C2'],
    'vintage_1': ['#3F3E3C', '#F2B53B', '#F4DFC0', '#F18274', '#D83F37'],
    'rainbow_1': ['#A8E6CF', '#DCEDC1', '#FFD3B6', '#FFAAA5', '#FF8B94'],
}


########################################################################################################


def plot_PCA_space_all_sets(pca, x, label_list, fig_name,
                            title=None, color_list=None, alpha_list=None,
                            legend=True, legend_outside=True, save_fig=True,
                            ):
    plt.figure(figsize=[6, 4.5])
    plt.rcParams["font.family"]=FONTFAMILY
    plt.xlabel("PC1 (" + str(round(pca.explained_variance_ratio_[0]*100, 1)) + "%)")
    plt.ylabel("PC2 (" + str(round(pca.explained_variance_ratio_[1]*100, 1)) + "%)")
    if title: plt.title(title, y=1.02)
    # plt.minorticks_on()
    # plt.xticks(minor=True)
    # plt.yticks(minor=True)
    plt.grid(linestyle='dotted', linewidth=0.8)

    for i in range(len(x)):
        if color_list:
            plt.scatter(x[i][:,0], x[i][:,1], label=label_list[i], color=color_list[i], alpha=alpha_list[i])
        else:
            plt.scatter(x[i][:,0], x[i][:,1], label=label_list[i])

    if legend:
        if legend_outside:
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, frameon=True)
        else:
            plt.legend(frameon=True)

    plt.subplots_adjust(top=0.95, bottom=0.15, left=0.15, right=0.925)
    if save_fig: plt.savefig('fig/PCA_{}.pdf'.format(fig_name))
    plt.show()
    plt.close()


########################################################################################################


def plot_hist(data_list, label_list, x_name, y_name, fig_name,
              title=None, x_lim=None, y_lim=None,
              x_ticks=None, rotate_x_ticks=False, y_ticks=None,
              fontweight='normal', lw=0.5,
              density=False, bins_=10, color_list=None,
              text=False, text_data=None,
              legend=False, legend_loc='upper right', legend_outside=False, save_fig=True,
              ):
    # plt.figure(figsize=[6, 4.5])
    plt.figure(figsize=[7.5, 4.5])
    plt.rcParams["font.family"]=FONTFAMILY
    if title: plt.title(title, y=1.02, fontweight=fontweight)
    plt.xlabel(x_name, fontweight=fontweight)
    plt.ylabel(y_name, fontweight=fontweight)
    if x_lim: plt.xlim(x_lim[0], x_lim[1])
    if y_lim: plt.ylim(y_lim[0], y_lim[1])
    # plt.minorticks_on()
    if x_ticks:
        if rotate_x_ticks:
            plt.xticks(x_ticks, rotation=rotate_x_ticks, fontweight=fontweight)
        else:
            plt.xticks(x_ticks, fontweight=fontweight)
    if y_ticks: plt.yticks(y_ticks, fontweight=fontweight)

    plt.grid(linestyle='dotted', linewidth=0.8)

    if (color_list and bins_):
        plt.hist(data_list, label=label_list, stacked=True, density=density, rwidth=0.8, color=color_list, bins=bins_, lw=lw)
    elif color_list:
        plt.hist(data_list, label=label_list, stacked=True, density=density, rwidth=0.8, color=color_list, lw=lw)
    elif bins_:
        plt.hist(data_list, label=label_list, stacked=True, density=density, rwidth=0.8, bins=bins_, lw=lw)
    else:
        plt.hist(data_list, label=label_list, stacked=True, density=density, rwidth=0.8, lw=lw)

    if text:
        for data in data_list:
            hist, edge = np.histogram(data, bins=bins_)
            for i in range(len(hist)):
                x = (edge[i]+edge[i+1])/2
                y = hist[i]
                plt.text(x, y, y, size=13, c='black', ha='center')
    if text_data:
        hist, edge = np.histogram(text_data, bins=bins_)
        for i in range(len(hist)):
            x = (edge[i]+edge[i+1])/2
            y = hist[i]
            plt.text(x, y, y, size=13, c='black', ha='center')

    if legend:
        if legend_outside:
            plt.legend(bbox_to_anchor=(1.05, 1), loc=legend_loc, borderaxespad=0, frameon=True)
        else:
            plt.legend(frameon=True, loc=legend_loc)

    plt.subplots_adjust(top=0.95, bottom=0.15, left=0.15, right=0.925)
    if save_fig: plt.savefig('fig/distribution_' + fig_name + '.pdf')
    plt.show()
    plt.close()


########################################################################################################


def plot_scatter(x_list, y_list, label_list, x_name, y_name, fig_name,
                 title=None, x_ticks=None, y_ticks=None, xlim=None, ylim=None,
                 fontweight='normal',
                 color_list=None, alpha_list=None, mark_list=None, s=None,
                 legend=True, legend_loc='upper left', legend_outside=False,
                 diagonal=False, plot_linear=None,
                 save_fig=True,
                 ):
    #     default: [6.4, 4.8]
    plt.figure(figsize=[6, 4.5])
    plt.rcParams["font.family"]=FONTFAMILY
    if title: plt.title(title, y=1.02, fontweight=fontweight)
    plt.xlabel(x_name, fontweight=fontweight)
    plt.ylabel(y_name, fontweight=fontweight)
    # plt.minorticks_on()
    if x_ticks: plt.xticks(x_ticks, fontweight=fontweight)
    else: plt.xticks(fontweight=fontweight)
    if y_ticks: plt.yticks(y_ticks, fontweight=fontweight)
    else: plt.yticks(fontweight=fontweight)
    plt.grid(linestyle='dotted', linewidth=0.8)
    if xlim:
        plt.xlim(xlim)
        plt.ylim(ylim)

    if diagonal:
        if x_ticks:
            plt.plot(x_ticks, y_ticks, color='dimgrey', linestyle="dashed")
        else:
            plt.plot(range(-9, -2), range(-9, -2), color='dimgrey', linestyle="dashed")

    for i in range(len(x_list)):
        if color_list:
            if mark_list:
                plt.scatter(x_list[i], y_list[i], label=label_list[i],
                            color=color_list[i], alpha=alpha_list[i], marker=mark_list[i], s=s)
            else:
                plt.scatter(x_list[i], y_list[i], label=label_list[i],
                            color=color_list[i], alpha=alpha_list[i], s=s)
        elif alpha_list:
            plt.scatter(x_list[i], y_list[i], label=label_list[i], alpha=alpha_list[i], s=s)
        else:
            plt.scatter(x_list[i], y_list[i], label=label_list[i], s=s)


    if legend:
        if legend_outside:
            plt.legend(bbox_to_anchor=(1.05, 1), loc=legend_loc, borderaxespad=0, frameon=True)
        else:
            plt.legend(frameon=True, loc=legend_loc)


    if plot_linear:
        a, b = plot_linear[0], plot_linear[1]
        plt.plot(x_ticks, [a*x+b for x in x_ticks], color='red', linestyle="dashed")

    plt.subplots_adjust(top=0.95, bottom=0.15, left=0.15, right=0.925)
    if save_fig: plt.savefig('fig/scatter_' + fig_name + '.pdf')
    plt.show()
    plt.close()


########################################################################################################


def plot_loss(loss_list, label_list, figname,
              log_scaled_y=False,
              title=None, loss_name=None,
              color_list=None,
              legend=True, legend_outside=False, save_fig=True,
              early_stop_epoch=None,
              ):

    plt.figure(figsize=[6, 4.5])
    plt.rcParams["font.family"]=FONTFAMILY
    if title: plt.title(title, y=1.02)
    plt.grid(linestyle='dotted', linewidth=0.8)

    plt.xlabel('Epoch')
    plt.ylabel(f'Loss ({loss_name})') if loss_name else plt.ylabel('Loss (mean squared error)')

    for i in range(len(loss_list)):
        if log_scaled_y:
            loss_list[i] = np.log(loss_list[i])
        if color_list:
            plt.plot(loss_list[i], label=label_list[i], color=color_list[i])
        else:
            plt.plot(loss_list[i], label=label_list[i])

    if early_stop_epoch:
        plt.axvspan(early_stop_epoch-0.5, early_stop_epoch+0.5, alpha=0.5, color='#E1EBF5')

    if legend:
        if legend_outside:
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, frameon=True)
        else:
            plt.legend(frameon=True, loc='upper right')

    plt.subplots_adjust(top=0.95, bottom=0.15, left=0.15, right=0.925)
    if save_fig: plt.savefig(f'fig/loss_{figname}.pdf')
    plt.show()
    plt.close()


########################################################################################################


def plot_line(data_list, label_list, x_name, y_name,
              color_list, marker_list, linestyle_list,
              fig_name=None, title=None,
              x_ticks=None, y_ticks=None, xlim=None, ylim=None,
              legend=True, legend_outside=False,
              save_fig=False,
              ):
    plt.figure(figsize=[6, 4.5])
    plt.rcParams["font.family"]=FONTFAMILY
    if title: plt.title(title, y=1.02)
    plt.grid(linestyle='dotted', linewidth=0.8)

    plt.xlabel(x_name)
    plt.ylabel(y_name)

    if x_ticks: plt.xticks(x_ticks[0], x_ticks[1], rotation=45)
    if y_ticks: plt.yticks(y_ticks)

    if xlim: plt.xlim(xlim)
    if ylim: plt.ylim(ylim)

    for i in range(len(data_list)):
        plt.plot(data_list[i], label=label_list[i], color=color_list[i],
                 marker=marker_list[i], markersize=10, linestyle=linestyle_list[i])

    if legend:
        if legend_outside:
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, frameon=True)
        else:
            plt.legend(frameon=True)

    plt.subplots_adjust(top=0.95, bottom=0.15, left=0.15, right=0.925)
    if save_fig: plt.savefig('fig/line_' + fig_name + '.pdf')
    plt.show()
    plt.close()



########################################################################################################
def plot_bar(name_list, height_list, label_list,
              color_list,
              height_list_2=None,
              figname=None,
              fontweight='normal',
              title=None, x_name=None, y_name=None, text=False,
              x_ticks=None, y_ticks=None, rotation=None, x_ticks_size=None,
              x_lim=None, y_lim=None, grid=True, spines=True,
              save_fig=False,
              legend=True, legend_outside=False,
              ):
    plt.figure(figsize=[6, 4.5])
    plt.rcParams["font.family"]=FONTFAMILY

    if title: plt.title(title, y=1.02, fontweight=fontweight)
    plt.xlabel(x_name, fontweight=fontweight)
    plt.ylabel(y_name, fontweight=fontweight)
    if x_ticks:
        if x_ticks_size:
            plt.xticks(x_ticks[0], x_ticks[1], rotation=rotation, fontweight=fontweight, fontsize=x_ticks_size)
        else:
            plt.xticks(x_ticks[0], x_ticks[1], rotation=rotation, fontweight=fontweight)
    if y_ticks: plt.yticks(y_ticks, fontweight=fontweight)
    else: plt.yticks(fontweight=fontweight)
    if x_lim: plt.xlim(x_lim[0], x_lim[1])
    if y_lim: plt.ylim(y_lim[0], y_lim[1])

    if grid:
        plt.grid(linestyle='dotted', linewidth=0.8)
    else:
        plt.grid(False)
    if not spines:
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['left'].set_color('black')
        plt.gca().spines['bottom'].set_color('black')


    if height_list_2:
        for name, height, height_2 in zip(name_list, height_list, height_list_2):
            plt.bar(name, height, align='edge', width=-0.3, label=label_list[0], color=color_list[0])
            plt.bar(name, height_2, align='edge', width=0.3, label=label_list[1], color=color_list[1])
    else:
        for name, height in zip(name_list, height_list):
            plt.bar(name, height, align='center', width=0.6, label=label_list[0], color=color_list[0])

        if text:
            for i in range(len(height)):
                x = list(range(len(height)))[i]
                y = height[i]
                plt.text(x, y, y, size=14, c='black', ha='center', fontweight='normal')


    if legend:
        if legend_outside:
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, frameon=True)
        else:
            plt.legend(frameon=True)

    plt.subplots_adjust(top=0.95, bottom=0.15, left=0.15, right=0.925)
    if save_fig: plt.savefig(f'fig/bar_{figname}.pdf')
    plt.show()
    plt.close()



########################################################################################################


def plot_barh(name_list, width_list, label_list, color_list,
              figname=None,
              title=None, x_name=None, y_name=None,
              x_ticks=None, y_ticks=None,
              x_lim=None, y_lim=None,
              save_fig=False,
              legend=False, legend_outside=False,
              ):
    plt.figure(figsize=[6, 4.5])
    plt.rcParams["font.family"]=FONTFAMILY
    if title: plt.title(title, y=1.02)
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    # plt.minorticks_on()
    if x_ticks: plt.xticks(x_ticks)
    if y_ticks: plt.yticks(y_ticks)
    if x_lim: plt.xlim(x_lim[0], x_lim[1])
    if y_lim: plt.ylim(y_lim[0], y_lim[1])
    plt.grid(linestyle='dotted', linewidth=0.8)
    for name, width in zip(name_list, width_list):
        plt.barh(name, width, label=label_list[0], color=color_list[0])

    if legend:
        if legend_outside:
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, frameon=True)
        else:
            plt.legend(frameon=True)

    plt.subplots_adjust(top=0.95, bottom=0.15, left=0.15, right=0.925)
    if save_fig: plt.savefig(f'fig/loss_{figname}.pdf')
    plt.show()
    plt.close()


########################################################################################################


def plot_heatmap(data,
                # 'Blues' 'Reds' 'Greens
                 cmap='Blues', annot=False, cbar=False, vmin=None, vmax=None,
                 non_tick_label=False,
                 x_name=None, y_name=None, title=None,
                 x_ticks=None, y_ticks=None, rotate_x_ticks=0, rotate_y_ticks=0,
                 ):
    plt.figure(figsize=[8, 6])
    plt.rcParams["font.family"]=FONTFAMILY
    if title: plt.title(title, y=1.02)
    if x_name: plt.xlabel(x_name)
    if y_name: plt.ylabel(y_name)
    if x_ticks: plt.xticks(x_ticks)
    if y_ticks: plt.yticks(y_ticks)
    if rotate_x_ticks: plt.xticks(rotation=rotate_x_ticks)
    if rotate_y_ticks: plt.yticks(rotation=rotate_y_ticks)

    if non_tick_label:
        plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)

    sns.heatmap(data, annot=annot, cmap=cmap, square=True, cbar=cbar)

    plt.subplots_adjust(top=0.95, bottom=0.15, left=0.15, right=0.925)
    plt.show()
    plt.close()


########################################################################################################

def linear_interpolate(color_start, color_end, alpha):
    """
    Linearly interpolate between two colors.

    :param color_start: Start color as a tuple (R, G, B).
    :param color_end: End color as a tuple (R, G, B).
    :param alpha: Interpolation factor (0-1).
    :return: Interpolated color as a tuple (R, G, B).
    """
    return tuple(start + alpha * (end - start) for start, end in zip(color_start, color_end))


########################################################################################################


def view_mol_2D(x, smarts=False, showindex=True, width=500, height=500,
                legend='', highlightAtoms=[], highlightAtomColors=[],
                fig_path=None):
    if ((type(x) == str) or (type(x) == np.str_)):
        mol = Chem.MolFromSmiles(x)
    else:
        mol = x

    if smarts:
        mol = Chem.MolFromSmarts(Chem.MolToSmarts(mol))
        Chem.SanitizeMol(mol)

    rdCoordGen.AddCoords(mol)

    view = rdMolDraw2D.MolDraw2DSVG(width, height)
    option = view.drawOptions()
    if showindex:
        option.addAtomIndices = True
    option.legendFontSize = 25
    # CHANGED 0.8 -> 1
    option.annotationFontScale = 0.6
    # enantiomer, chirality
    # option.addStereoAnnotation = True

    view.DrawMolecule(rdMolDraw2D.PrepareMolForDrawing(mol),
                      highlightAtoms=highlightAtoms,
                      highlightAtomColors=highlightAtomColors,
                      legend=legend)

    # コンテナをファイナライズ
    view.FinishDrawing()

    # コンテナに書き込んだデータを取り出す
    svg = view.GetDrawingText()

    if fig_path:
        with open(fig_path, "w") as f:
            f.write(svg)
        f.close()

    # データを描画
    display(SVG(svg.replace('svg:,','')))




def convert_to_color_dict(aDict, intensity=0.7):
    """
    Takes a dictionary type as input and returns a dictionary type with the value of the dictionary type converted to a blue-red colormap.
    """

    # intensity must be [0.1, 1.0]
    intensity = min(max(intensity, 0.1), 1.0)

    vals = np.array(list(aDict.values()))

    # scale values to [-1,1]
    vals = vals / np.max(np.abs(vals))
    # convert [-1,1] to [0,1]
    vals = (vals + 1) / 2
    # intensity calibration
    vals = vals * intensity

    # # scale values to [0,1]
    # # WARNING だめ、青も出てくる
    # vals = (vals - np.min(vals)) / (np.max(vals) - np.min(vals))

    color_dict = {k:cm.bwr(v)[:3] for k,v in zip(aDict.keys(), vals)}
    return color_dict
