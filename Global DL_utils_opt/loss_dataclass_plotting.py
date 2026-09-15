#### 📉 Loss Collection & Visualization ########################################
#
# - 🧠 This module provides tools for collecting, organizing, and visualizing
#   loss values during the training of deep learning models.
#
# - 🪜 Losses are stored hierarchically:
#       🗃️ Fold  →  📈 Epoch  →  📦 Batch
#
# - 📊 The visualization functions enable analysis of:
#       🏋️ Training and 🧪 validation loss trends
#       📈 Model convergence and stability
#       🔄 Differences across training runs or folds

import os
import sys

from dataclasses import dataclass, field
from typing import List, Optional
import plotly.graph_objects as go
import numpy as np

# Get the current working directory. Either get injected value from globals or
# default to os.getcwd(). This enables easy testing
CWD = globals().get("CWD", os.getcwd())

# Define the path to the global utilities directory. Either get injected value
# from globals or default. This enables easy testing
GLOBAL_UTILS_DIR = globals().get(
    "GLOBAL_UTILS_DIR",
    os.path.normpath(
        os.path.join(CWD, "..", "Global utils_opt", "Pipeline-V2")
    ),
)

# Append this directory to the system path
sys.path.append(GLOBAL_UTILS_DIR)
# Import the global utilities
from global_utils import *


#### 📉 LossManager ############################################################
#
# - 🗂️ The LossManager provides a hierarchical and organized way to store the loss
#   values produced during model training. 
# - 📚  Losses are grouped according to the training procedure 
#   (cross-validation training, cross-validation validation, or full-dataset training) 
#   and are stored at multiple levels
#   - 🪜 Fold  →  📈 Epoch  →  📦 Batch
#
# - 🔍 This hierarchy allows loss values to be accessed at different levels of
#   granularity, making it straightforward to analyze training dynamics,
#   convergence, and stability.
# - 🔄 The manager automatically creates the required Fold, Epoch, and Batch
#   objects as training progresses, providing a clean interface for recording
#   and retrieving losses for visualization and performance evaluation.


@dataclass
class Batch:
    """
    Dataclass representing the loss of a single batch. The decorator @dataclass
    is used to automatically generate special methods like __init__ and __repr__.
    The loss attribute is optional, allowing for the possibility that a batch may
    not have a loss value assigned at the time of creation.
    """

    loss: Optional[float] = None

    def __repr__(self):
        return str(self.loss)


@dataclass
class Epoch:
    """
    Dataclass representing the loss of an epoch, which consists of multiple batches.
    The loss attribute is optional, allowing for the possibility that an epoch may
    not have a loss value assigned at the time of creation. The batch_losses
    property is defined to calculate and return a list of losses for all batches
    within the epoch that have a loss value assigned. The property is an computed
    attribute which is not stored in the instance but is calculated on-the-fly
    when accessed.
    """

    loss: Optional[float] = None
    batch: List[Batch] = field(default_factory=list)

    @property
    def batch_losses(self):
        return [b.loss for b in self.batch if b.loss is not None]

    def __repr__(self):
        return str(self.loss)


@dataclass
class Fold:
    """
    Dataclass representing the loss of a fold, which consists of multiple epochs.
    The loss attribute is optional, allowing for the possibility that a fold may
    not have a loss value assigned at the time of creation. The epoch_losses property is
    defined to calculate and return a list of losses for all epochs within the
    fold that have a loss value assigned.
    """

    loss: Optional[float] = None
    epoch: List[Epoch] = field(default_factory=list)

    @property
    def epoch_losses(self):
        return [e.loss for e in self.epoch if e.loss is not None]

    def __repr__(self):
        return str(self.loss)


@dataclass
class HierarchicalLosses:
    """
    Dataclass representing the hierarchical structure of losses for different
    kinds of training and validation processes. The kind attribute specifies the
    type of loss (e.g., 'CV_training', 'CV_validation', 'full_training'), and
    the folds attribute is a list of Fold instances that represent the losses
    for each fold in the training/validation process. The fold_losses property is
    defined to calculate and return a list of losses for all folds.
    """

    loss: Optional[float] = None
    fold: List[Fold] = field(default_factory=list)

    @property
    def fold_losses(self):
        return [f.loss for f in self.fold if f.loss is not None]

    def __repr__(self):
        return str(self.loss)


class LossManager:
    """
    Class to elegantly keep track of the different losses during the training
    and validation process, organized in a hierarchical structure of batches,
    epochs, and folds for different kinds of training and validation processes.
    The losses are stored in a dictionary where the keys are the kinds of
    training/validation (e.g., 'CV_training', 'CV_validation', 'Full_training')
    and the values are instances of the HierarchicalLosses dataclass that
    contain the folds, epochs, and batches with their respective losses, e.g.
    class.losses["CV_training"].fold[0].epoch[0].batch[0].loss to access
    the loss of the first batch of the first epoch of the first fold of the
    CV training process. To access the losses for all batches in the first
    epoch of the first fold of the CV training process, you can use
    class.losses["CV_training"].fold[0].epoch[0].batch_losses.
    """

    def __init__(self):
        """
        Initialize the LossManager with empty HierarchicalLosses for each kind of
        training/validation process.

        """
        self.CV_training = HierarchicalLosses()
        self.CV_validation = HierarchicalLosses()
        self.Full_training = HierarchicalLosses()

    def _current_kind(self, kind: str) -> HierarchicalLosses:
        """
        Get the kind of training/validation process specified by the kind parameter.
        The kind parameter is an instance variable.
        """
        if not hasattr(self, kind):
            raise ValueError(
                "Invalid kind. Must be "
                "'CV_training', "
                "'CV_validation', "
                "or 'Full_training'."
            )

        return getattr(self, kind)

    def _current_fold(self, kind: str) -> Fold:
        """
        Get the current fold for the specified kind of training/validation process.
        If no fold exists, create a new one and return it.
        """
        # Get the kind of training/validation process
        current_kind = self._current_kind(kind)

        # Check if there are any folds in the group,
        # if not create a new fold and return it
        if not current_kind.fold:
            current_kind.fold.append(Fold())

        # Return the current fold
        return current_kind.fold[-1]

    def _current_epoch(self, kind: str) -> Epoch:
        """
        Get the current epoch for the specified kind of training/validation process.
        If no epoch exists in the current fold, create a new one and return it.
        """

        # Get the current fold
        current_fold = self._current_fold(kind)

        # Check if there are any epochs in the fold, if not create a new epoch and return it
        if not current_fold.epoch:
            current_fold.epoch.append(Epoch())

        # Return the current epoch
        return current_fold.epoch[-1]

    def add_batch(self, kind: str, loss: float):
        """
        Add the loss of a batch to the current epoch of the current fold. The loss
        is added as an instance of the Batch dataclass to the batches list of
        the current epoch.
        """

        # Get the current epoch for the specified kind of training/validation process
        current_epoch = self._current_epoch(kind)
        # Add the loss of the batch to the current epoch as an instance of the
        # Batch dataclass
        current_epoch.batch.append(Batch(loss=loss))

    def add_epoch(self, kind: str, loss: Optional[float] = None):
        """
        Add the loss of an epoch to the current fold. The loss is assigned to the
        current epoch, and then a new epoch is created for future batch losses.
        """

        # Get the current fold for the specified kind of training/validation process
        current_fold = self._current_fold(kind)

        # Assign the loss to the current epoch
        current_fold.epoch[-1].loss = loss

        # Create a new epoch for future batch losses
        current_fold.epoch.append(Epoch())

    def add_fold(self, kind: str, loss: Optional[float] = None):
        """
        Add the loss of a fold to the specified kind of training/validation process.
        The loss is assigned to the current fold, and then a new fold is created
        for future epochs and batches.
        """

        # Get the current kind of training/validation process
        current_kind = self._current_kind(kind)

        # Assign the loss to the current fold
        current_kind.fold[-1].loss = loss

        # Create a new fold for future epochs and batches
        current_kind.fold.append(Fold())
        
        
        
#### 📉 Loss Plotting ##########################################################
#
# - 🎨 The loss plotting module provides a flexible and interactive way to
#   visualize model training progress using Plotly.
#
# - 🗂️ Losses stored in the LossManager can be visualized according to different
#   training procedures:
#       - 🔄 Cross-validation training
#       - 🏋️ Full-dataset training
#
# - 🪜 Loss values can be displayed at different levels of granularity:
#       - 🗃️ Fold   →  📈 Epoch   →  📦 Batch
#
# - 📊 Supported visualizations include:
#       - 🏋️ Training and 🧪 validation loss curves
#       - 📈 Mean convergence trends across cross-validation folds
#       - 🧬 Individual fold learning paths with reduced opacity
#       - 🚧 Epoch boundaries and validation intervals for batch-level analysis
#
# - 🛠️ The module automatically handles:
#       - 📏 Loss curves with different lengths
#       - 🧮 Mean loss calculation across folds
#       - 🎨 Consistent colors, markers, and hover information
#       - 🖱️ Interactive Plotly visualization
#


def _add_loss_trace(
    fig: go.Figure,
    x: list,
    y: list,
    name: str,
    color: str,
    dash: str = "solid",
    marker_symbol: str = "circle",
    opacity: float = 1.0,
    showlegend: bool = True,
    legendgroup: str = None,
    hovertemplate: str = None,
    lines_markers: str = "lines+markers",
) -> None:
    """
    Helper function to add a loss trace with consistent styling and hover information.

    Parameters:
        - fig: The Plotly figure to which the trace will be added.
        - x: The x-values for the trace (e.g., epoch numbers, batch numbers, fold numbers).
        - y: The y-values for the trace (e.g., loss values).
        - name: The name of the trace to be displayed in the legend.
        - color: The color of the line and markers for the trace.
        - dash: The dash style of the line (e.g., "solid", "dot", "dash"). Default is "solid".
        - marker_symbol: The symbol for the markers (e.g., "circle", "diamond"). Default is "circle".
        - opacity: The opacity of the trace (between 0 and 1). Default is 1.0 (fully opaque).
        - showlegend: Whether to show this trace in the legend. Default is True.
        - legendgroup: The name of the legend group for this trace. Traces in
                          the same group will be toggled together in the legend. Default is None.
        - hovertemplate: The hover template for the trace, allowing customization
                         of the hover information. Default is None.
        - lines_markers: A string indicating the mode for the trace. Valid options
                        are "lines", "markers", and "lines+markers". Default is "lines+markers".
    """

    # Add a scatter trace to the figure with the specified parameters for styling
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode=lines_markers,
            name=name,
            line=dict(color=color, dash=dash),
            marker=dict(color=color, symbol=marker_symbol),
            opacity=opacity,
            showlegend=showlegend,
            legendgroup=legendgroup,
            hovertemplate=hovertemplate,
        )
    )


def _mean_curve(loss_lists: list) -> tuple:
    """
    Helper function to compute the mean curve across multiple lists of losses,

    Parameters:
        - loss_lists: A list of lists, where each inner list contains loss values
                        (e.g., per epoch or per batch) for a fold or run. The inner
                        lists can have different lengths.
    """

    # Get the maximum length of the loss lists to create a matrix for mean
    # calculation
    max_len = max(len(losses) for losses in loss_lists)

    # Create a matrix where each row corresponds to a loss list, and pad shorter
    # lists with NaN to allow the calculation of the mean across folds without
    # being affected by different lengths
    loss_matrix = np.array(
        [losses + [np.nan] * (max_len - len(losses)) for losses in loss_lists],
        dtype=float,
    )

    return np.nanmean(loss_matrix, axis=0), max_len


def _flatten_batch_losses(fold: Fold) -> list:
    """
    Helper function to flatten the batch losses across all epochs in a fold into a single list.

    Parameters:
        - fold: A Fold object that contains the epoch information and batch losses.
    """

    return [loss for epoch in fold.epoch for loss in epoch.batch_losses]


def _train_add_epoch_boundaries(fig: go.Figure, fold: Fold, color: str = "red") -> None:
    """
    Helper function to add vertical lines and annotations to indicate the
    boundaries between epochs in a fold.

    Parameters:
        - fig: The Plotly figure to which the epoch boundary lines and annotations will be added.
        - fold: A Fold object that contains the epoch information and batch losses.
                The function will iterate through the epochs in the fold to determine
                where to place the vertical lines and annotations for epoch boundaries.
        - color: The color of the vertical lines and annotations. Default is "red".
    """
    # Only add a line if there are more than 2 epochs in the fold, as a single
    # epoch does not require a boundary
    if len(fold.epoch) <= 2:
        return

    # Initialize a running batch counter to keep track of the cumulative number
    # of batches as we iterate through the epochs
    running_batch = 0

    # Iterate through each epoch in the fold to determine where to place vertical lines
    # and annotations for epoch boundaries.
    for epoch_idx, epoch in enumerate(fold.epoch):

        # Skip epochs that do not have batch losses
        if not epoch.batch_losses:
            continue

        # Update the running batch counter by adding the number of batches in the
        # current epoch
        running_batch += len(epoch.batch_losses)

        # Calculate the x-coordinate for the vertical line to indicate the epoch
        # boundary. We add 0.5 to place the line between the last batch of the
        # current epoch and the first batch of the next epoch
        separator_x = running_batch + 0.5

        # Add the vertical line and annotation to the figure to indicate the
        # epoch boundary.
        fig.add_vline(
            x=separator_x,
            line=dict(color=color, dash="dash"),
        )

        fig.add_annotation(
            x=separator_x,
            y=1.005,
            xref="x",
            yref="paper",
            text=str(epoch_idx + 1),
            showarrow=False,
            font=dict(color=color),
            yanchor="bottom",
            xanchor="center",
            name=f"Epoch Boundary",
        )


def _val_add_epoch_boundaries(
    fig: go.Figure, val_start: int, val_max_batches: int, color: str = "red"
) -> None:
    """
    Helper function to add vertical lines and annotations to indicate the start
    and end of the  validation batches across all folds.

    Parameters:
        - fig: The Plotly figure to which the validation boundary lines and annotations will be added.
        - val_start: The starting x-coordinate for the validation batches, which is
                        calculated based on the maximum number of training batches and
                        a small offset to avoid overlap with the training batches.
        - val_max_batches: The maximum number of validation batches across all folds,
                            which is used to calculate the ending x-coordinate for the
                            validation batches.
        - color: The color of the vertical lines and annotations. Default is "red".
    """

    # Calculate the left and right x-coordinates for the vertical lines to indicate
    # the start and end of the validation batches. We subtract and add 0.5
    # to place the lines between the boundaries.
    validation_left = val_start - 0.5
    validation_right = val_start + val_max_batches - 0.5

    # Add the vertical lines and annotation to the figure to indicate the start and
    # end of the validation batches across all folds.
    fig.add_vline(
        x=validation_left,
        line=dict(color=color, dash="dash", width=2),
    )

    fig.add_vline(
        x=validation_right,
        line=dict(color=color, dash="dash", width=2),
    )

    fig.add_annotation(
        x=(validation_left + validation_right) / 2,
        y=1.005,
        xref="x",
        yref="paper",
        text="Validation",
        showarrow=False,
        font=dict(color=color),
        yanchor="bottom",
        xanchor="center",
    )


def _plot_cv_fold(
    loss_manager: LossManager, title: str, lines_markers: str
) -> go.Figure:
    """
    Helper function to visualize the cross-validation training and validation
    losses per fold, with separate traces for training and validation losses.

    Parameters:
        - loss_manager: An instance of the LossManager class that contains the
                        fold_losses for both CV training and CV validation, which
                        will be used to create the traces for the figure.
        - title: An optional string specifying the title of the plot.
        - lines_markers: A string indicating the mode for the traces in the plot.
    """

    # Get the training and validation fold losses from the loss manager for
    # cross-validation
    train_losses = loss_manager.CV_training.fold_losses
    val_losses = loss_manager.CV_validation.fold_losses

    fig = go.Figure()

    # Add the loss trace for the training
    _add_loss_trace(
        fig,
        list(range(1, len(train_losses) + 1)),
        train_losses,
        name="Training Loss" + "   ",
        color="blue",
        marker_symbol="circle",
        hovertemplate=("Fold: %{x}<br>" "CV Training Loss: %{y:.4f}" "<extra></extra>"),
        lines_markers=lines_markers,
    )

    # Add the loss trace for the validation
    _add_loss_trace(
        fig,
        list(range(1, len(val_losses) + 1)),
        val_losses,
        name="Validation Loss" + "   ",
        color="orange",
        marker_symbol="circle",
        hovertemplate=(
            "Fold: %{x}<br>" "CV Validation Loss: %{y:.4f}" "<extra></extra>"
        ),
        lines_markers=lines_markers,
    )

    fig.update_layout(
        # title=title if title is not None else "Cross-Validation Loss per Fold",
        xaxis_title="Fold",
        yaxis_title="Loss",
        legend_title="Loss Type" + "   ",
    )

    return fig


def _plot_cv_epoch(
    loss_manager: LossManager, title: str, lines_markers: str
) -> go.Figure:
    """
    Helper function to visualize the cross-validation training and validation
    losses per epoch, with separate traces for training and validation losses,
    as well as mean curves across folds and individual fold curves.

    Parameters:
        - loss_manager: An instance of the LossManager class that contains the
                        epoch_losses for both CV training and CV validation, which
                        will be used to create the traces for the figure.
        - title: An optional string specifying the title of the plot.
        - lines_markers: A string indicating the mode for the traces in the plot.
    """
    # Get the training and validation epoch losses from the loss manager for
    # cross-validation
    train_losses = [fold.epoch_losses for fold in loss_manager.CV_training.fold]
    val_losses = [fold.epoch_losses for fold in loss_manager.CV_validation.fold]

    # Calculate the mean curves across folds for training and validation losses
    mean_train, max_epochs = _mean_curve(train_losses)
    mean_val, max_epochs_val = _mean_curve(val_losses)

    fig = go.Figure()

    # Add the mean loss traces for training epochs
    _add_loss_trace(
        fig,
        np.arange(1, max_epochs + 1),
        mean_train,
        name="Mean Training Loss" + "   ",
        color="blue",
        marker_symbol="diamond",
        hovertemplate=(
            "Epoch: %{x}<br>" "Mean Training Loss: %{y:.4f}" "<extra></extra>"
        ),
        lines_markers=lines_markers,
    )

    # Add the mean loss traces for validation epochs
    _add_loss_trace(
        fig,
        np.arange(max_epochs + 2, max_epochs + 2 + max_epochs_val),
        mean_val,
        name="Mean Validation Loss" + "   ",
        color="orange",
        marker_symbol="diamond",
        hovertemplate=(
            "Epoch: %{x}<br>" "Mean Validation Loss: %{y:.4f}" "<extra></extra>"
        ),
        lines_markers=lines_markers,
    )

    # Iterate through each fold's training and validation losses to add
    # individual fold traces with lower opacity
    for idx, losses in enumerate(train_losses):

        # Add a trace for the training losses of the current fold with lower opacity
        _add_loss_trace(
            fig,
            np.arange(1, len(losses) + 1),
            losses,
            name="Training Loss" + "   ",
            color="blue",
            dash="dot",
            opacity=0.8,
            showlegend=(idx == 0),
            legendgroup="training_folds",
            hovertemplate=("Epoch: %{x}<br>" "Loss: %{y:.4f}" "<extra></extra>"),
            lines_markers=lines_markers,
        )

        # Add a trace for the validation losses of the current fold with lower opacity
        _add_loss_trace(
            fig,
            np.arange(max_epochs + 2, max_epochs + 2 + len(val_losses[idx])),
            val_losses[idx],
            name="Validation Loss" + "   ",
            color="orange",
            dash="dot",
            opacity=0.8,
            showlegend=(idx == 0),
            legendgroup="validation_folds",
            hovertemplate=("Epoch: %{x}<br>" "Loss: %{y:.4f}" "<extra></extra>"),
            lines_markers=lines_markers,
        )

    fig.update_layout(
        # title = title if title is not None else "Cross-Validation Training and Validation Loss per Epoch",
        xaxis_title="Epoch",
        yaxis_title="Loss",
        legend_title="Loss Type" + "   ",
    )

    return fig


def _plot_cv_batch(
    loss_manager: LossManager, title: str, lines_markers: str
) -> go.Figure:
    """
    Helper function to visualize the cross-validation training and validation
    losses per batch, with separate traces for training and validation losses,
    as well as mean curves across folds and individual fold curves. Add a small
    offset to the validation batches on the x-axis for better readability.

    Parameters:
        - loss_manager: An instance of the LossManager class that contains the
                        batch_losses for both CV training and CV validation, which
                        will be used to create the traces for the figure.
        - title: An optional string specifying the title of the plot.
        - lines_markers: A string indicating the mode for the traces in the plot.
    """

    # Get the training and validation batch losses from the loss manager for
    # cross-validation and flatten them across epochs for each fold
    train_loss_lists = [
        _flatten_batch_losses(fold) for fold in loss_manager.CV_training.fold
    ]
    val_loss_lists = [
        _flatten_batch_losses(fold) for fold in loss_manager.CV_validation.fold
    ]

    # Calculate the mean curves across folds for training and validation batch losses
    train_mean, train_max_batches = _mean_curve(train_loss_lists)
    val_mean, val_max_batches = _mean_curve(val_loss_lists)

    # Get the maximum batch length across all epochs and folds for training to determine
    # where to place the validation batches on the x-axis without overlap
    train_max_batch_length = max(
        len(epoch.batch_losses)
        for fold in loss_manager.CV_training.fold
        for epoch in fold.epoch
        if epoch.batch_losses
    )

    # Calculate the starting and ending x-coordinates for the validation batches
    # to ensure they are placed after all training batches without overlap
    val_start = train_max_batches + int(0.05 * train_max_batch_length) + 1
    val_end = val_start + val_max_batches

    # Create x-values for the training and validation losses, ensure that the
    # the validation batches start after the training batches to avoid overlap
    # on the x-axis
    train_batch_idx = np.arange(1, train_max_batches + 1)
    val_batch_idx = np.arange(val_start, val_end)

    fig = go.Figure()

    # Add the mean loss traces for training batches
    _add_loss_trace(
        fig,
        train_batch_idx,
        train_mean,
        name="Mean Training Loss" + "   ",
        color="blue",
        marker_symbol="diamond",
        hovertemplate=(
            "Batch: %{x}<br>" "Mean Training Loss: %{y:.4f}" "<extra></extra>"
        ),
        lines_markers=lines_markers,
    )

    # Add the mean loss traces for validation batches
    _add_loss_trace(
        fig,
        val_batch_idx,
        val_mean,
        name="Mean Validation Loss" + "   ",
        color="orange",
        marker_symbol="diamond",
        hovertemplate=(
            "Batch: %{x}<br>" "Mean Validation Loss: %{y:.4f}" "<extra></extra>"
        ),
        lines_markers=lines_markers,
    )

    # Iterate through each fold's training and validation batch losses to add
    for idx, fold in enumerate(loss_manager.CV_training.fold):

        # Get the flattened batch losses for the current fold
        train_losses = _flatten_batch_losses(fold)

        # Add a trace for the training losses of the current fold with lower opacity
        _add_loss_trace(
            fig,
            train_batch_idx,
            train_losses,
            name="Training Loss" + "   ",
            color="blue",
            dash="dot",
            opacity=0.7,
            showlegend=(idx == 0),
            legendgroup="training_folds",
            hovertemplate=(
                "Batch: %{x}<br>" "Training Loss: %{y:.4f}" "<extra></extra>"
            ),
            lines_markers=lines_markers,
        )

        # Add vertical lines to indicate epoch boundaries for the training
        # batches of the first fold
        if idx == 0:
            _train_add_epoch_boundaries(fig, fold)

    # Iterate through each fold's training and validation batch losses to add
    for idx, fold in enumerate(loss_manager.CV_validation.fold):

        # Get the flattened batch losses for the current fold
        val_losses = _flatten_batch_losses(fold)

        # Add a trace for the validation losses of the current fold with lower opacity
        _add_loss_trace(
            fig,
            val_batch_idx,
            val_losses,
            name="Validation Loss" + "   ",
            color="orange",
            dash="dot",
            opacity=0.7,
            showlegend=(idx == 0),
            legendgroup="validation_folds",
            hovertemplate=(
                "Batch: %{x}<br>" "Validation Loss: %{y:.4f}" "<extra></extra>"
            ),
            lines_markers=lines_markers,
        )

    # Add vertical lines to indicate the start and end of the validation batches across all folds
    _val_add_epoch_boundaries(fig, val_start, val_max_batches, color="red")

    # Add dummy traces for the validation region and epoch boundary to include
    # them in the legend
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="lines",
            line=dict(color="red", dash="dash"),
            name="Validation Boundary" + "   ",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="lines",
            line=dict(color="red", dash="dash"),
            name="Epoch Boundary" + "   ",
        )
    )

    fig.update_layout(
        # title= title if title is not None else "Cross-Validation Training and Validation Loss per Batch",
        xaxis_title="Batch",
        yaxis_title="Loss",
        legend_title="Loss Type" + "   ",
        margin=dict(t=120),
        hovermode="x unified",
        xaxis=dict(
            range=[
                int(-0.05 * max(val_batch_idx)),
                int(max(val_batch_idx) + 0.05 * max(val_batch_idx)),
            ]
        ),
    )

    return fig


def _plot_full_epoch(
    loss_manager: LossManager, title: str, lines_markers: str
) -> go.Figure:
    """
    Helper function to visualize the full data training loss per epoch.

    Parameters:
        - loss_manager: An instance of the LossManager class that contains the
                        epoch_losses for full data training, which will be used to
                        create the trace for the figure.
        - title: An optional string specifying the title of the plot.
        - lines_markers: A string indicating the mode for the traces in the plot.
    """

    # Get the training epoch losses from the loss manager for full data training
    losses = loss_manager.Full_training.fold[0].epoch_losses

    fig = go.Figure()

    # Add the loss trace for the full data training epochs
    _add_loss_trace(
        fig,
        np.arange(1, len(losses) + 1),
        losses,
        name="Training Loss" + "   ",
        color="blue",
        hovertemplate=("Epoch: %{x}<br>" "Training Loss: %{y:.4f}" "<extra></extra>"),
        lines_markers=lines_markers,
    )

    fig.update_layout(
        # title=title if title is not None else "Full Data Training Loss per Epoch",
        xaxis_title="Epoch",
        yaxis_title="Loss",
        legend_title="Loss Type" + "   ",
    )

    return fig


def _plot_full_batch(
    loss_manager: LossManager, title: str, lines_markers: str
) -> go.Figure:
    """
    Helper function to visualize the full data training loss per batch.

    Parameters:
        - loss_manager: An instance of the LossManager class that contains the
                        batch_losses for full data training, which will be used to
                        create the trace for the figure.
        - title: An optional string specifying the title of the plot.
        - lines_markers: A string indicating the mode for the traces in the plot.
    """

    # Get the first fold of the full data training from the loss manager to access the batch losses
    fold = loss_manager.Full_training.fold[0]

    # Get the flattened batch losses across all epochs in the fold to create a
    # single list of batch losses for the full data training
    losses = _flatten_batch_losses(fold)

    fig = go.Figure()

    # Add the loss trace for the full data training batches
    _add_loss_trace(
        fig,
        np.arange(1, len(losses) + 1),
        losses,
        name="Training Loss" + "   ",
        color="blue",
        opacity=0.8,
        hovertemplate=("Batch: %{x}<br>" "Training Loss: %{y:.4f}" "<extra></extra>"),
        lines_markers=lines_markers,
    )

    # Add vertical lines to indicate epoch boundaries for the training batches
    _train_add_epoch_boundaries(fig, fold)

    # Add a dummy trace for the epoch boundary to include it in the legend if
    # there are more than 2 epochs
    if len(fold.epoch) > 2:
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="lines",
                line=dict(color="red", dash="dash"),
                name="Epoch Boundary" + "   ",
            )
        )

    fig.update_layout(
        # title=title if title is not None else "Full Data Training Loss per Batch",
        xaxis_title="Batch",
        yaxis_title="Loss",
        legend_title="Loss Type" + "   ",
        margin=dict(t=110),
        xaxis=dict(
            range=[
                int(-0.05 * len(losses) + 1),
                int(len(losses) + 1 + 0.05 * len(losses) + 1),
            ]
        ),
    )

    return fig


def plot_loss(
    loss_manager: LossManager,
    training_type: str = "CV",
    agg_type: str = "epoch",
    lines_markers: str = "lines+markers",
    title: str = None,
    save_fig=None,
):
    """
    Main function to plot the training and validation losses for the hierarchical
    transformer supporting different combinations of training types (CV vs Full Data) and
    aggregation types (Fold vs Epoch vs Batch).

    Parameters:
        - loss_manager: An instance of the LossManager class that contains the
                        training and validation losses for both CV and Full Data training,
                        which will be used to create the traces for the figure.
        - training_type: A string indicating the type of training to visualize.
                         Valid options are "CV" for cross-validation and "Full Data" for
                         full data training. Default is "CV".
        - agg_type: A string indicating the level of aggregation for the losses to visualize.
                        Valid options are "Fold", "Epoch", and "Batch". Default is "epoch".
        - lines_markers: A string indicating the mode for the traces in the plot.
                         Valid options are "lines", "markers", and "lines+markers".
                            Default is "lines+markers".
        - title: An optional string specifying the title of the plot. If None, a default
                    title will be generated based on the training_type and agg_type. Default is None.
        - save_fig: An optional string specifying the file path to save the generated
                    figure as an HTML file. If None, the figure will not be saved. Default is None.

    """

    # Ensure lines and markers is a valid option
    if lines_markers not in ["lines", "markers", "lines+markers"]:
        raise ValueError(
            f"Invalid lines_markers: {lines_markers}. Choose from 'lines', 'markers', or 'lines+markers'."
        )

    # Call the appropriate plotting function based on the specified training_type
    # and agg_type
    if training_type == "CV":

        if agg_type == "Fold":
            fig = _plot_cv_fold(loss_manager, title=title, lines_markers=lines_markers)

        elif agg_type == "Epoch":
            fig = _plot_cv_epoch(loss_manager, title=title, lines_markers=lines_markers)

        elif agg_type == "Batch":
            fig = _plot_cv_batch(loss_manager, title=title, lines_markers=lines_markers)
        else:
            raise ValueError(
                f"Invalid agg_type for CV: {agg_type} choose from 'Fold', 'Epoch', or 'Batch'."
            )

    elif training_type == "Full Data":

        if agg_type == "Epoch":
            fig = _plot_full_epoch(
                loss_manager, title=title, lines_markers=lines_markers
            )

        elif agg_type == "Batch":
            fig = _plot_full_batch(
                loss_manager, title=title, lines_markers=lines_markers
            )
        else:
            raise ValueError(
                f"Invalid agg_type for Full Data: {agg_type} choose from 'Epoch' or 'Batch'."
            )
    else:
        raise ValueError(
            f"Invalid training_type: {training_type} choose from 'CV' or 'Full Data'."
        )

    # Set the x-axis tick format based on the aggregation type.
    # For everything except 'Batch', we use a linear tick mode with a step of 1.
    if agg_type != "Batch":
        fig.update_xaxes(
            tickmode="linear",
            tick0=1,
            dtick=1,
            tickformat="d",
        )
    # Set the x-axis tick format for 'Batch' aggregation type to display integer
    # values.
    else:
        fig.update_xaxes(
            tick0=1,
            tickformat="d",
        )

    # Save the figure if a filename is provided
    if save_fig:
        save_fig_formats(fig, save_fig)

    return fig
