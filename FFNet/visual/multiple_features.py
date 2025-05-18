from morphopy.computation import feature_presentation as fp
from morphopy import neurontree
import morphopy
print(dir(morphopy.neurontree.utils))


filename = '/data/suncl/data/20_classes/swc_data/basket/030905005.CNG.swc'

neutree = neurontree.load_swc(filename)
dp = fp.compute_density_maps(neutree)
statics = fp.compute_morphometric_statistics(neutree)
persistence = fp.get_persistence(neutree)
#plot
fp.plot_density_maps(dp['x'])