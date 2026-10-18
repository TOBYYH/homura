//! What callers can't do with the parameter groups of an optimizer, as `compile_fail`
//! documentation tests from a caller's crate. The state of a parameter is kept by its position
//! in the groups, so only the options of a group can be changed (R64 of review/M2.md).

/// The changes the examples below rely on, which compile: the options of a group, through
/// `options_mut`, and reading the groups.
///
/// ```
/// use homura::nn::Parameter;
/// use homura::optim::{Optimizer, ParamGroup, Sgd, SgdOptions};
///
/// let (a, b) = (Parameter::new(homura::ones([2])), Parameter::new(homura::ones([2])));
/// let mut opt = Sgd::new([a], 0.1);
/// opt.add_param_group(ParamGroup::new([b], SgdOptions { lr: 0.2, ..*opt.defaults() }));
/// opt.options_mut(1).lr = 0.01;
/// let groups = opt.param_groups();
/// assert_eq!(groups[1].options.lr, 0.01);
/// assert_eq!(groups[0].params().len(), 1);
/// ```
///
/// There is no mutable access to the groups:
///
/// ```compile_fail,E0599
/// # use homura::nn::Parameter;
/// # use homura::optim::{Optimizer, Sgd};
/// let mut opt = Sgd::new([Parameter::new(homura::ones([2]))], 0.1);
/// opt.param_groups_mut().swap(0, 0);
/// ```
///
/// The groups that `param_groups` lends can't be reordered,
///
/// ```compile_fail,E0596
/// # use homura::nn::Parameter;
/// # use homura::optim::{Optimizer, ParamGroup, Sgd};
/// let mut opt = Sgd::new([Parameter::new(homura::ones([2]))], 0.1);
/// opt.add_param_group(ParamGroup::new([], *opt.defaults()));
/// opt.param_groups().swap(0, 1);
/// ```
///
/// replaced,
///
/// ```compile_fail,E0594
/// # use homura::nn::Parameter;
/// # use homura::optim::{Optimizer, ParamGroup, Sgd};
/// let mut opt = Sgd::new([Parameter::new(homura::ones([2]))], 0.1);
/// let other = ParamGroup::new([Parameter::new(homura::ones([2]))], *opt.defaults());
/// opt.param_groups()[0] = other;
/// ```
///
/// or changed:
///
/// ```compile_fail,E0594
/// # use homura::nn::Parameter;
/// # use homura::optim::{Optimizer, Sgd};
/// let mut opt = Sgd::new([Parameter::new(homura::ones([2]))], 0.1);
/// opt.param_groups()[0].options.lr = 0.01;
/// ```
///
/// The parameters of a group are private, also of a group the caller owns:
///
/// ```compile_fail,E0616
/// # use homura::nn::Parameter;
/// # use homura::optim::{ParamGroup, SgdOptions};
/// # let options = SgdOptions { lr: 0.1, momentum: 0.0, dampening: 0.0, weight_decay: 0.0,
/// #     nesterov: false, maximize: false };
/// let mut group = ParamGroup::new([Parameter::new(homura::ones([2]))], options);
/// group.params.clear();
/// ```
struct ParamGroupErrors;
