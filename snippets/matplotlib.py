import matplotlib.pyplot as plt
from matplotlib import cm, colors


# Get number of rows and columns of plt.Figure
# -- src: https://stackoverflow.com/a/64537128
fig, axes = plt.subplots(nrows=1, ncols=4) ## shape=(4,)
gs = axes[0].get_gridspec()
gs.nrows  # return  1
gs.ncols  # returns 4


# How colormapping works in matplotlib
## My examples:
## Demo1: Normalize (linear) 
# At init, vmin and vmax are not set yet:
normalizer = colors.Normalize(
    vmin=None,
    vmax=None,
    clip=False
)
print('vmin, vmax: ', normalizer.vmin, normalizer.vmax)
print('vmin, vmax are set? :', normalizer.scaled())

# at first normalization call using this class, its vmin and vmax will be set
# to the min and max of the input arr __call__ was called:
arr = [-2., -1., 0., 1., 2.]
normed_arr = normalizer(arr)
print('vmin, vmax: ', normalizer.vmin, normalizer.vmax)
print('vmin, vmax are set? :', normalizer.scaled())
assert normalizer.vmin == min(arr) and normalizer.vmax == max(arr)

print('arr: ', arr)
print('normed_arr: ', normed_arr)


## Demo2: Normalize (linear) 
# At init, vmin and vmax are specified by user.
# w/ clip=False
normalizer = colors.Normalize(
    vmin=-1,
    vmax=1,
    clip=False
)
print('vmin, vmax: ', normalizer.vmin, normalizer.vmax)
print('vmin, vmax are set? :', normalizer.scaled())

# at first normalization call using this class, its vmin and vmax will be set
# to the min and max of the input arr __call__ was called:
arr = [-2., -1., 0., 1., 2.]
normed_arr = normalizer(arr)
print('vmin, vmax: ', normalizer.vmin, normalizer.vmax)
print('vmin, vmax are set? :', normalizer.scaled())

print('arr: ', arr)
print('normed_arr: ', normed_arr)
print("Note that if clip=False, then the normalizer does not enforce clipping on the normed values "
      "to [0.,1.]: ie., if normed value is out of range [0,1], it just gives out those values. ")


## Demo3: Normalize (linear) 
# At init, vmin and vmax are specified by user.
# w/ clip=True
normalizer = colors.Normalize(
    vmin=-1,
    vmax=1,
    clip=True
)
print('vmin, vmax: ', normalizer.vmin, normalizer.vmax)
print('vmin, vmax are set? :', normalizer.scaled())

# at first normalization call using this class, its vmin and vmax will be set
# to the min and max of the input arr __call__ was called:
arr = [-2., -1., 0., 1., 2.]
normed_arr = normalizer(arr)
print('vmin, vmax: ', normalizer.vmin, normalizer.vmax)
print('vmin, vmax are set? :', normalizer.scaled())

print('arr: ', arr)
print('normed_arr: ', normed_arr)


## Demo4-a: LogNorm
# At init, vmin and vmax are not specified.
# w/ clip=False
normalizer = colors.LogNorm(
    vmin=None,
    vmax=None,
    clip=False
)
print('vmin, vmax: ', normalizer.vmin, normalizer.vmax)
print('vmin, vmax are set? :', normalizer.scaled())

# at first normalization call using this class, its vmin and vmax will be set
# to the min and max of the input arr __call__ was called:
arr = [-2., -1., 0., 1., 2.]
normed_arr = normalizer(arr)
print('vmin, vmax: ', normalizer.vmin, normalizer.vmax) #0,2?
print('vmin, vmax are set? :', normalizer.scaled())

print()
print('arr: ', arr)
print('normed_arr: ', normed_arr)


## Demo4-b: LogNorm
# At init, vmin and vmax are not specified.
# w/ clip=False
normalizer = colors.LogNorm(
    vmin=None,
    vmax=None,
    clip=False
)
print('vmin, vmax: ', normalizer.vmin, normalizer.vmax)
print('vmin, vmax are set? :', normalizer.scaled())
print('---')

# arr = [-10., -1., 0., 1, 10., 100., 1000]
arr = [-10., -1., 0., 0.001, 1, 10., 100., 1000]
# arr = [-10., -1., 0., 1, 10., 100., 1000]
normed_arr = normalizer(arr)
print('vmin, vmax: ', normalizer.vmin, normalizer.vmax) #0,1000?
print('vmin, vmax are set? :', normalizer.scaled())
print('---')

print('arr: ', arr)
print('normed_arr: ', normed_arr)


## Demo 5: LogNorm
# At init, vmin and vmax are not specified.
# w/ clip=False
# Note: vmin and vmax properties of a Normalize object is set either at init time
#      or at the first __call__ time. 
# In particular, any subsequent __call__ (with diff. data array to normalize)
#     does not change the vmin and vmax values.

normalizer = colors.LogNorm(
    vmin=None,
    vmax=None,
    clip=False
)
print('vmin, vmax: ', normalizer.vmin, normalizer.vmax)
print('vmin, vmax are set? :', normalizer.scaled())

# at first normalization call using this class, its vmin and vmax will be set
# to the min and max of the input arr __call__ was called:
arr = [-2., -1., 0., 1., 2.]
normed_arr = normalizer(arr)
print('vmin, vmax: ', normalizer.vmin, normalizer.vmax) #0,2?
print('vmin, vmax are set? :', normalizer.scaled())

print('arr: ', arr)
print('normed_arr: ', normed_arr)


print('===')
print("Note: Normalize object's vmin and vmax are set either at init time or at the first call. "
      "-- ie, any subsequent __call__ does not change vmin and vmax!! ")

arr = [-10., -1., 0., 10., 100., 1000]
normed_arr = normalizer(arr)
print('vmin, vmax: ', normalizer.vmin, normalizer.vmax) #0,1000?
print('vmin, vmax are set? :', normalizer.scaled())

print('arr: ', arr)
print('normed_arr: ', normed_arr)
