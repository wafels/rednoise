#
# Utilities to plot out details of step 1 of the analysis
#
import matplotlib.pyplot as plt
import analysis_get_data


#
# Make a plot with the locations of the regions
#
def plot_regions(image, regions, filepath):
    """

    :param image:
    :param regions:
    :param filepath:
    :return:
    """
    plt.close('all')
    fig, ax = plt.subplots()
    ret = image.plot()
    # for patch in patches:
    for region in sorted(regions.keys()):
        patch = regions[region]["patch"]
        label_offset = regions[region]["label_offset"]
        ax.add_patch(patch)
        llxy = patch.get_xy()
        height = patch.get_height()
        width = patch.get_width()
        plt.text(llxy[0] + width + label_offset['x'],
                 llxy[1] + height + label_offset['y'],
                 patch.get_label(),
                 bbox=dict(facecolor='w', alpha=0.5))

    # Get the sunspot outline
    sunspot_outline = analysis_get_data.sunspot_outline()
    ax.add_collection(analysis_get_data.rotate_sunspot_outline(sunspot_outline[0], sunspot_outline[1], image.date, linewidth=[1]))
    ax.autoscale_view()

    #plt.show()
    if filepath is not None:
        plt.savefig(filepath)
    else:
        plt.show()
    return None


def plot_regions_hsr2015(image, filepath):
    """
    Plots the regions as required by the HSR 2015.

    :param image:
    :param filepath:
    :return:
    """
    # Cut down the map
    image = analysis_get_data.hsr2015_map(image)

    plt.close('all')
    fig, ax = plt.subplots()
    ret = image.plot()

    # Get the sunspot outline
    sunspot_outline = analysis_get_data.sunspot_outline()
    polygon = sunspot_outline[0]
    sunspot_date = sunspot_outline[1]
    print 'Sunspot date = %s' % sunspot_date
    print 'Image date = %s' % image.date
    ax.add_collection(analysis_get_data.rotate_sunspot_outline(polygon, sunspot_date, image.date, linewidth=[1]))
    ax.autoscale_view()

    #plt.show()
    if filepath is not None:
        plt.savefig(filepath)
    else:
        plt.show()
    return None
