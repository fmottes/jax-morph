import jax.numpy as np
import matplotlib.pyplot as plt

# set global properties of plots
plt.rcParams.update({"font.size": 18})


def draw_network_shells(W, labels, shells, pruning_threshold=0.0, ax=None, style=None):
    """Draw a network with improved aesthetics.

    Args:
        W: Weight matrix
        eps: Threshold for edge weights
        labels: Node labels dictionary
        shells: List of lists defining node arrangement in shells. First shell is the innermost (output layer), last shell is the outermost (input layer).
        ax: Matplotlib axis (optional)
        style: Dict of visual parameters (optional)
    """
    import networkx as nx
    import matplotlib.patches as mpatches

    # Default style parameters
    default_style = {
        "node_size": 900,
        # "node_colors": ["#2ecc71", "#3498db", "#e74c3c"][::-1],  # color palette
        "node_colors": ["indianred", "steelblue", "seagreen"],
        "node_alpha": 0.5,
        "edge_width_scale": 2,
        "edge_curve": 0.23,
        "edge_margin": 20,
        "font_size": 10,
        "font_color": "#2c3e50",
        # "pos_edge_color": "darkgreen",  # Green for positive edges
        # "neg_edge_color": "darkred",  # Red for negative edges
        "pos_edge_color": "black",
        "neg_edge_color": "black",
        "node_edgecolor": "black",
        "node_linewidth": 0.5,
        "node_label_offset": 0.05,
    }

    # Update style with user parameters if provided
    if style is not None:
        default_style.update(style)

    # Create directed graph
    G = nx.from_numpy_array(W, create_using=nx.DiGraph)

    # Get edges and weights
    edge_act = []
    edge_inh = []
    W_act = []
    W_inh = []

    for u, v, w in G.edges(data=True):
        weight = w["weight"]
        if weight > pruning_threshold:
            edge_act += [(u, v)]
            W_act += [weight]
        elif weight < -pruning_threshold:
            edge_inh += [(u, v)]
            W_inh += [weight]

    W_act = np.array(W_act)
    W_inh = np.array(W_inh)

    # Handle empty arrays by using -inf for empty max operations
    w_scale = np.maximum(
        np.max(W_act) if W_act.size > 0 else -np.inf,
        np.max(-W_inh) if W_inh.size > 0 else -np.inf,
    )
    # If both arrays were empty, set scale to 1 to avoid -inf
    if w_scale == -np.inf:
        w_scale = 1.0

    # Get active nodes
    nodelist = [
        n
        for n in G.nodes
        if (n in np.array(edge_act).flatten() or n in np.array(edge_inh).flatten())
    ]

    # Filter shells to only include active nodes
    shells = [[n for n in shell if n in nodelist] for shell in shells]

    # Create layout
    pos = nx.shell_layout(G, shells)
    # pos = nx.planar_layout(G)

    # Assign colors to nodes based on shell membership
    node_colors = (
        [default_style["node_colors"][2]] * len(shells[2])
        + [default_style["node_colors"][1]] * len(shells[1])
        + [default_style["node_colors"][0]] * len(shells[0])
    )

    if ax is None:
        ax = plt.gca()

    # Draw nodes
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=nodelist,
        node_size=default_style["node_size"],
        node_color=node_colors,
        edgecolors=default_style["node_edgecolor"],
        linewidths=default_style["node_linewidth"],
        alpha=default_style["node_alpha"],
        ax=ax,
    )

    # Draw positive edges
    if len(W_act) > 0:
        edge_widths = W_act / w_scale * default_style["edge_width_scale"]
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=edge_act,
            edge_color=default_style["pos_edge_color"],
            width=edge_widths,
            alpha=0.7,
            arrows=True,
            arrowsize=30,
            node_size=default_style["node_size"],
            arrowstyle="-|>",
            connectionstyle=f'arc3,rad={default_style["edge_curve"]}',
            min_source_margin=default_style["edge_margin"],
            min_target_margin=default_style["edge_margin"],
            ax=ax,
        )

    # Draw negative edges
    if len(W_inh) > 0:
        edge_widths = -W_inh / w_scale * default_style["edge_width_scale"]
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=edge_inh,
            edge_color=default_style["neg_edge_color"],
            style="--",
            arrowstyle=mpatches.ArrowStyle.BarAB(widthA=0.0, widthB=0.9),
            width=edge_widths,
            alpha=0.7,
            arrows=True,
            arrowsize=10,
            node_size=default_style["node_size"],
            connectionstyle=f'arc3,rad={-default_style["edge_curve"]}',
            min_source_margin=default_style["edge_margin"],
            min_target_margin=default_style["edge_margin"],
            ax=ax,
        )

    # Draw labels
    # Split nodes into input/hidden/output based on shells
    input_nodes = shells[-1]  # Left shell
    output_nodes = shells[0]  # Right shell
    hidden_nodes = [n for shell in shells[1:-1] for n in shell]  # Middle shells

    # Draw labels for hidden nodes centered on nodes
    nx.draw_networkx_labels(
        G,
        pos,
        labels={n: labels[n] for n in hidden_nodes},
        font_size=default_style["font_size"],
        font_color=default_style["font_color"],
        font_weight="bold",
        ax=ax,
    )

    # Draw labels for input/output nodes below nodes
    pos_below = {
        n: (pos[n][0], pos[n][1] - default_style["node_label_offset"])
        for n in input_nodes + output_nodes
    }
    nx.draw_networkx_labels(
        G,
        pos_below,
        labels={n: labels[n] for n in input_nodes + output_nodes},
        font_size=default_style["font_size"],
        font_color=default_style["font_color"],
        font_weight="bold",
        ax=ax,
    )

    # Style the plot
    ax.set_facecolor("white")
    ax.axis("off")

    return plt.gcf(), ax


def draw_gene_network(network, ax=None, style=None):
    """Draw a gene regulatory network visualization from a GeneNetwork instance.

    Args:
        network: GeneNetwork instance to visualize
        eps: Threshold for edge weights (default: 0.1)
        ax: Matplotlib axis (optional)
        style: Dict of visual parameters (optional)
    """

    # Default style parameters
    default_style = {
        "node_size": 1000,
        "node_colors": [
            "#2ecc71",
            "#3498db",
            "#e74c3c",
        ],  # Input, Hidden, Output colors
        "node_alpha": 0.5,
        "edge_width_scale": 2,
        "edge_curve": 0.23,
        "font_size": 10,
        "font_color": "#2c3e50",
        "pos_edge_color": "#2ecc71",  # Green for positive edges
        "neg_edge_color": "#e74c3c",  # Red for negative edges
    }

    if style is not None:
        default_style.update(style)

    # Create node labels using the network's attributes
    labels = {}
    idx = 0

    # Input nodes
    for i, field in enumerate(network.input_fields):
        labels[idx] = f"IN_{field}"
        idx += 1

    # Hidden nodes
    for i in range(network.hidden_size):
        labels[idx] = f"H{i+1}"
        idx += 1

    # Output nodes
    for i, field in enumerate(network.output_fields):
        labels[idx] = f"OUT_{field}"
        idx += 1

    # Create shells for layout using correct sizes
    shells = [
        list(range(network.input_size)),  # Input layer
        list(
            range(network.input_size, network.input_size + network.hidden_size)
        ),  # Hidden layer
        list(
            range(
                network.input_size + network.hidden_size,
                network.input_size + network.hidden_size + network.output_size,
            )
        ),  # Output layer
    ][::-1]

    # Use the existing draw_network_shells function with the interaction matrix
    # Note: interaction matrix is already in the correct orientation (i -> j)
    return draw_network_shells(
        network.interaction_matrix,
        labels=labels,
        shells=shells,
        ax=ax,
        style=style,
    )


def OLD_draw_network(W, eps, labels, shells, ax=None):

    import networkx as nx

    G = nx.from_numpy_array(W.T, create_using=nx.DiGraph)
    edge_act = [(u, v) for u, v, w in G.edges(data=True) if w["weight"] > eps]
    edge_in = [(u, v) for u, v, w in G.edges(data=True) if w["weight"] < -eps]

    W_act = np.array(
        [w["weight"] for u, v, w in G.edges(data=True) if w["weight"] > eps]
    )
    W_in = np.array(
        [w["weight"] for u, v, w in G.edges(data=True) if w["weight"] < -eps]
    )

    nodelist = [
        n
        for n in G.nodes
        if (n in np.array(edge_act).flatten() or n in np.array(edge_in).flatten())
    ]

    # keep only nodes in nodelist in shells
    shells = [[n for n in shell if n in nodelist] for shell in shells]

    # pos = nx.spring_layout(G)
    pos = nx.shell_layout(G, shells)

    node_colors = (
        ["seagreen"] * len(shells[2])
        + ["steelblue"] * len(shells[1])
        + ["indianred"] * len(shells[0])
    )

    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=nodelist,
        node_size=480,
        node_color=node_colors,
        edgecolors="black",
        alpha=0.95,
        ax=ax,
    )
    if len(W_act) > 0:
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=edge_act,
            alpha=W_act / np.max(W_act),
            node_size=480,
            arrows=True,
            nodelist=nodelist,
            connectionstyle="arc3,rad=0.1",
            ax=ax,
        )
    if len(W_in) > 0:
        nx.draw_networkx_edges(
            G,
            pos,
            style="--",
            edgelist=edge_in,
            alpha=-W_in / np.max(-W_in),
            node_size=480,
            arrows=True,
            nodelist=nodelist,
            connectionstyle="arc3,rad=0.1",
            ax=ax,
        )
    nx.draw_networkx_labels(
        G,
        pos,
        labels={n: labels[n] for n in nodelist},
        font_size=8,
        font_color="black",
        ax=ax,
    )

    # remove axis
    # plt.gca().set_axis_off()

    # plt.title("Gene network");
    return ax


def draw_circles_ctype(state, ax=None, cm=plt.cm.coolwarm, grid=False, **kwargs):

    if ax is None:
        ax = plt.axes()

    if "alpha" in kwargs:
        alpha = kwargs["alpha"]
        kwargs.pop("alpha")
    else:
        alpha = 0.5

    alive_cells = np.squeeze(state.celltype.sum(1) > 0)

    # only usable for two cell types
    # color = cm(np.float32(state.celltype-1)[alive_cells])

    if state.celltype.shape[1] > 1:
        color_levels = (
            state.celltype @ 2 ** np.arange(state.celltype.shape[1]) - 1
        ) / (2 ** (state.celltype.shape[1] - 1) - 1)
    else:
        color_levels = state.celltype - 1

    color = cm(color_levels)

    for cell, radius, c in zip(
        state.position[alive_cells],
        state.radius[alive_cells].squeeze(),
        color[alive_cells],
    ):
        circle = plt.Circle(cell, radius=radius, color=c, alpha=alpha, **kwargs)
        ax.add_patch(circle)

    # calculate ax limits
    xmin = np.min(state.position[:, 0][alive_cells])
    xmax = np.max(state.position[:, 0][alive_cells])

    ymin = np.min(state.position[:, 1][alive_cells])
    ymax = np.max(state.position[:, 1][alive_cells])

    max_coord = max([xmax, ymax]) + 3
    min_coord = min([xmin, ymin]) - 3

    ax.set_xlim(min_coord, max_coord)
    ax.set_ylim(min_coord, max_coord)

    # scale x and y in the same way
    ax.set_aspect("equal", adjustable="box")

    # white bg color for ax
    ax.set_facecolor([1, 1, 1])

    if grid:
        ax.grid(alpha=0.2)
    else:
        # remove axis spines
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)

        ax.set_xticks([])
        ax.set_yticks([])

    background_color = [56 / 256] * 3
    plt.gcf().patch.set_facecolor(background_color)
    plt.gcf().patch.set_alpha(0)

    plt.gcf().set_size_inches(8, 8)

    return plt.gcf(), ax


# Visualization of multiple cell types
def draw_circles_ctypes(state, ax=None, cm=plt.cm.coolwarm, grid=False, **kwargs):

    if ax is None:
        ax = plt.axes()

    if "alpha" in kwargs:
        alpha = kwargs["alpha"]
        kwargs.pop("alpha")
    else:
        alpha = 0.5

    alive_cells = np.squeeze(state.celltype.sum(1) > 0)
    color_levels = (np.argmax(state.celltype, axis=-1) + 1) / state.celltype.shape[1]
    color = cm(color_levels)

    for cell, radius, c in zip(
        state.position[alive_cells],
        state.radius[alive_cells].squeeze(),
        color[alive_cells],
    ):
        circle = plt.Circle(cell, radius=radius, color=c, alpha=alpha, **kwargs)
        ax.add_patch(circle)

    # calculate ax limits
    xmin = np.min(state.position[:, 0][alive_cells])
    xmax = np.max(state.position[:, 0][alive_cells])

    ymin = np.min(state.position[:, 1][alive_cells])
    ymax = np.max(state.position[:, 1][alive_cells])

    max_coord = max([xmax, ymax]) + 3
    min_coord = min([xmin, ymin]) - 3

    ax.set_xlim(min_coord, max_coord)
    ax.set_ylim(min_coord, max_coord)

    # scale x and y in the same way
    ax.set_aspect("equal", adjustable="box")

    # white bg color for ax
    ax.set_facecolor([1, 1, 1])

    if grid:
        ax.grid(alpha=0.2)
    else:
        # remove axis spines
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)

        ax.set_xticks([])
        ax.set_yticks([])

    background_color = [56 / 256] * 3
    plt.gcf().patch.set_facecolor(background_color)
    plt.gcf().patch.set_alpha(0)

    plt.gcf().set_size_inches(8, 8)

    return plt.gcf(), ax


def draw_circles_chem(
    state,
    chem=0,
    colorbar=True,
    ax=None,
    cm=None,
    grid=False,
    labels=False,
    edges=False,
    cm_edges=plt.cm.coolwarm,
    max_val=None,
    **kwargs,
):

    if ax is None:
        ax = plt.axes()

    if "alpha" in kwargs:
        alpha = kwargs["alpha"]
        kwargs.pop("alpha")
    else:
        alpha = 0.5

    alive_cells = np.squeeze(state.celltype.sum(1) > 0)

    chemical = state.chemical[:, chem][alive_cells]

    if max_val is None:
        max_val = chemical.max()

    chemical = (chemical + 1e-20) / (max_val + 1e-20)

    # only usable for two cell types
    if cm is None:
        if 0 == chem:
            cm = plt.cm.YlGn
        elif 1 == chem:
            cm = plt.cm.BuPu
        else:
            cm = plt.cm.coolwarm

    color = cm(chemical)

    if edges:
        # only usable for two cell types
        ct_color = cm_edges(np.float32(state.celltype - 1)[alive_cells])

        for cell, radius, c, ctc in zip(
            state.position[alive_cells],
            state.radius[alive_cells].squeeze(),
            color,
            ct_color,
        ):
            circle = plt.Circle(
                cell, radius=radius, fc=c, ec=ctc, lw=2, alpha=alpha, **kwargs
            )
            ax.add_patch(circle)

    else:
        for i, (cell, radius, c) in enumerate(
            zip(state.position[alive_cells], state.radius[alive_cells].squeeze(), color)
        ):
            circle = plt.Circle(cell, radius=radius, fc=c, alpha=alpha, **kwargs)
            ax.add_patch(circle)
            if labels:
                ax.text(
                    *cell,
                    str(i),
                    horizontalalignment="center",
                    verticalalignment="center",
                )

    # show colorbar
    if colorbar:
        # sm = plt.cm.ScalarMappable(cmap=cm, norm=plt.Normalize(vmin=state.chemical[:,chem][alive_cells].min(), vmax=state.chemical[:,chem][alive_cells].max()))
        sm = plt.cm.ScalarMappable(
            cmap=cm,
            norm=plt.Normalize(vmin=0, vmax=state.chemical[:, chem][alive_cells].max()),
        )

        sm._A = []
        cbar = plt.colorbar(sm, ax=ax, fraction=0.05, alpha=0.5)  # rule of thumb
        cbar.set_label("Conc. Chem. " + str(chem), labelpad=20)

    # calculate ax limits
    xmin = np.min(state.position[:, 0][alive_cells])
    xmax = np.max(state.position[:, 0][alive_cells])

    ymin = np.min(state.position[:, 1][alive_cells])
    ymax = np.max(state.position[:, 1][alive_cells])

    max_coord = max([xmax, ymax]) + 3
    min_coord = min([xmin, ymin]) - 3

    ax.set_xlim(min_coord, max_coord)
    ax.set_ylim(min_coord, max_coord)

    # scale x and y in the same way
    ax.set_aspect("equal", adjustable="box")

    # white bg color for ax
    ax.set_facecolor([1, 1, 1])

    if grid:
        ax.grid(alpha=0.2)
    else:
        # remove axis spines
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)

        ax.set_xticks([])
        ax.set_yticks([])

    background_color = [56 / 256] * 3
    plt.gcf().patch.set_facecolor(background_color)
    plt.gcf().patch.set_alpha(0)

    plt.gcf().set_size_inches(8, 8)

    return plt.gcf(), ax


def draw_circles_stress(
    state,
    colorbar=True,
    ax=None,
    cm=None,
    grid=False,
    labels=False,
    edges=False,
    cm_edges=plt.cm.coolwarm,
    **kwargs,
):

    if ax is None:
        ax = plt.axes()

    if "alpha" in kwargs:
        alpha = kwargs["alpha"]
        kwargs.pop("alpha")
    else:
        alpha = 0.5

    alive_cells = np.squeeze(state.celltype.sum(1) > 0)

    stress = state.mechanical_stress[alive_cells]
    stress = (stress - stress.min() + 1e-20) / (stress.max() - stress.min() + 1e-20)

    # only usable for two cell types
    if cm is None:
        cm = plt.cm.coolwarm

    color = cm(stress)

    if edges:
        # only usable for two cell types
        ct_color = cm_edges(np.float32(state.celltype - 1)[alive_cells])

        for cell, radius, c, ctc in zip(
            state.position[alive_cells],
            state.radius[alive_cells].squeeze(),
            color,
            ct_color,
        ):
            circle = plt.Circle(
                cell, radius=radius, fc=c, ec=ctc, lw=2, alpha=alpha, **kwargs
            )
            ax.add_patch(circle)

    else:
        for i, (cell, radius, c) in enumerate(
            zip(state.position[alive_cells], state.radius[alive_cells].squeeze(), color)
        ):
            circle = plt.Circle(cell, radius=radius, fc=c, alpha=alpha, **kwargs)
            ax.add_patch(circle)
            if labels:
                ax.text(
                    *cell,
                    str(i),
                    horizontalalignment="center",
                    verticalalignment="center",
                )

    # show colorbar
    if colorbar:
        sm = plt.cm.ScalarMappable(
            cmap=cm,
            norm=plt.Normalize(
                vmin=state.mechanical_stress[alive_cells].min(),
                vmax=state.mechanical_stress[alive_cells].max(),
            ),
        )
        sm._A = []
        cbar = plt.colorbar(sm, ax=ax, fraction=0.05, alpha=0.5)  # rule of thumb
        cbar.set_label("Mech. Stress", labelpad=20)

    # calculate ax limits
    xmin = np.min(state.position[:, 0][alive_cells])
    xmax = np.max(state.position[:, 0][alive_cells])

    ymin = np.min(state.position[:, 1][alive_cells])
    ymax = np.max(state.position[:, 1][alive_cells])

    max_coord = max([xmax, ymax]) + 3
    min_coord = min([xmin, ymin]) - 3

    ax.set_xlim(min_coord, max_coord)
    ax.set_ylim(min_coord, max_coord)

    # scale x and y in the same way
    ax.set_aspect("equal", adjustable="box")

    # white bg color for ax
    ax.set_facecolor([1, 1, 1])

    if grid:
        ax.grid(alpha=0.2)
    else:
        # remove axis spines
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)

        ax.set_xticks([])
        ax.set_yticks([])

    background_color = [56 / 256] * 3
    plt.gcf().patch.set_facecolor(background_color)
    plt.gcf().patch.set_alpha(0)

    plt.gcf().set_size_inches(8, 8)

    return plt.gcf(), ax


def draw_circles_division(
    state,
    probability=False,
    colorbar=True,
    ax=None,
    cm=plt.cm.coolwarm,
    grid=False,
    labels=False,
    edges=False,
    cm_edges=plt.cm.coolwarm,
    **kwargs,
):

    if ax is None:
        ax = plt.axes()

    if "alpha" in kwargs:
        alpha = kwargs["alpha"]
        kwargs.pop("alpha")
    else:
        alpha = 0.5

    alive_cells = np.squeeze(state.celltype.sum(1) > 0)

    division = state.division[alive_cells]
    division = (division - division.min() + 1e-20) / (
        division.max() - division.min() + 1e-20
    )

    color = cm(division)

    if edges:
        # only usable for two cell types
        ct_color = cm_edges(np.float32(state.celltype - 1)[alive_cells])

        for cell, radius, c, ctc in zip(
            state.position[alive_cells],
            state.radius[alive_cells].squeeze(),
            color,
            ct_color,
        ):
            circle = plt.Circle(
                cell, radius=radius, fc=c, ec=ctc, lw=2, alpha=alpha, **kwargs
            )
            ax.add_patch(circle)

    else:
        #
        for i, (cell, radius, c) in enumerate(
            zip(state.position[alive_cells], state.radius[alive_cells].squeeze(), color)
        ):
            circle = plt.Circle(cell, radius=radius, fc=c, alpha=alpha, **kwargs)
            ax.add_patch(circle)
            if labels:
                ax.text(
                    *cell,
                    str(i),
                    horizontalalignment="center",
                    verticalalignment="center",
                )

    # show colorbar
    if colorbar:
        if probability:
            division = state.division[alive_cells] / (
                state.division[alive_cells].sum() + 1e-20
            )
            sm = plt.cm.ScalarMappable(
                cmap=cm, norm=plt.Normalize(vmin=division.min(), vmax=division.max())
            )
            sm._A = []
            cbar_text = "Division Probability"
        else:
            sm = plt.cm.ScalarMappable(
                cmap=cm,
                norm=plt.Normalize(
                    vmin=state.division[alive_cells].min(),
                    vmax=state.division[alive_cells].max(),
                ),
            )
            sm._A = []
            cbar_text = "Division Propensity"

        cbar = plt.colorbar(sm, ax=ax, fraction=0.05, alpha=0.5)  # rule of thumb
        cbar.set_label(cbar_text, labelpad=20)

    # calculate ax limits
    xmin = np.min(state.position[:, 0][alive_cells])
    xmax = np.max(state.position[:, 0][alive_cells])

    ymin = np.min(state.position[:, 1][alive_cells])
    ymax = np.max(state.position[:, 1][alive_cells])

    max_coord = max([xmax, ymax]) + 3
    min_coord = min([xmin, ymin]) - 3

    ax.set_xlim(min_coord, max_coord)
    ax.set_ylim(min_coord, max_coord)

    # scale x and y in the same way
    ax.set_aspect("equal", adjustable="box")

    # white bg color for ax
    ax.set_facecolor([1, 1, 1])

    if grid:
        ax.grid(alpha=0.2)
    else:
        # remove axis spines
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)

        ax.set_xticks([])
        ax.set_yticks([])

    background_color = [56 / 256] * 3
    plt.gcf().patch.set_facecolor(background_color)
    plt.gcf().patch.set_alpha(0)

    plt.gcf().set_size_inches(8, 8)

    return plt.gcf(), ax


def draw_circles(
    state,
    state_values,
    min_val=None,
    max_val=None,
    min_coord=None,
    max_coord=None,
    ax=None,
    cm=plt.cm.coolwarm,
    grid=False,
    plt_cbar=True,
    cbar_title=None,
    **kwargs,
):

    if ax is None:
        ax = plt.axes()

    if "alpha" in kwargs:
        alpha = kwargs["alpha"]
        kwargs.pop("alpha")
    else:
        alpha = 0.5

    alive_cells = np.squeeze(state.celltype.sum(1) > 0)

    state_values = np.float32(state_values)[alive_cells]

    if min_val is None:
        # state_values = (state_values-state_values.min()+1e-20)/(state_values.max()-state_values.min()+1e-20)
        state_values = state_values
        min_val, max_val = state_values.min(), state_values.max()
    else:
        state_values = (state_values - min_val + 1e-20) / (max_val - min_val + 1e-20)

    # only usable for two cell types
    color = cm(state_values)
    for cell, radius, c in zip(
        state.position[alive_cells], state.radius[alive_cells].squeeze(), color
    ):
        circle = plt.Circle(cell, radius=radius, fc=c, alpha=alpha, **kwargs)
        ax.add_patch(circle)

    # calculate ax limits
    xmin = np.min(state.position[:, 0][alive_cells])
    xmax = np.max(state.position[:, 0][alive_cells])

    ymin = np.min(state.position[:, 1][alive_cells])
    ymax = np.max(state.position[:, 1][alive_cells])

    if min_coord is None:
        max_coord = max([xmax, ymax]) + 3
        min_coord = min([xmin, ymin]) - 3

    ax.set_xlim(min_coord, max_coord)
    ax.set_ylim(min_coord, max_coord)

    # scale x and y in the same way
    ax.set_aspect("equal", adjustable="box")

    # white bg color for ax
    ax.set_facecolor([1, 1, 1])

    if grid:
        ax.grid(alpha=0.2)
    else:
        # remove axis spines
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)

        ax.set_xticks([])
        ax.set_yticks([])

    sm = plt.cm.ScalarMappable(cmap=cm, norm=plt.Normalize(vmin=min_val, vmax=max_val))
    sm._A = []
    if plt_cbar:
        cbar = plt.colorbar(sm, ax=ax, fraction=0.05, alpha=0.5)  # rule of thumb
        if cbar_title is not None:
            cbar.set_label(cbar_title, labelpad=20)

    background_color = [56 / 256] * 3
    plt.gcf().patch.set_facecolor(background_color)
    plt.gcf().patch.set_alpha(0)

    plt.gcf().set_size_inches(8, 8)

    return plt.gcf(), ax


###################################################################################################
#                                                                                                 #
#                                   3D VISUALIZATION FUNCTIONALITIES                              #
#                                                                                                 #
###################################################################################################


# %matplotlib widget # add to notebook for interactive 3D plots


def draw_spheres(
    state,
    color_field="chemical",
    color_index=0,
    colorbar=True,
    ax=None,
    cm=None,
    grid=False,
    labels=False,
    max_val=None,
    elev=70,
    azim=-80,
    **kwargs,
):
    """
    Draw spheres in 3D space.

    color_field=None draws all spheres in a predefined color.
    """

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = plt.gcf()

    if "alpha" in kwargs:
        alpha = kwargs["alpha"]
        kwargs.pop("alpha")
    else:
        alpha = 0.6

    alive_cells = np.squeeze(state.celltype.sum(1) > 0)

    if color_field == "division":
        raise Warning(
            "Use draw_spheres_division for specialized visualization of division rates"
        )
    elif color_field is None:
        color_data = np.ones_like(state.celltype[:, 0])[alive_cells] * 0.4
        colorbar = False
        max_val = 1.0
    else:
        color_data = getattr(state, color_field)[:, color_index][alive_cells]

    if max_val is None:
        max_val = color_data.max()

    color_data = color_data / (max_val + 1e-20)

    # Default colormap
    if cm is None:
        cm = plt.cm.BuPu

    color = cm(color_data)

    for i, (cell, radius, c) in enumerate(
        zip(state.position[alive_cells], state.radius[alive_cells].squeeze(), color)
    ):
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x = radius * np.outer(np.cos(u), np.sin(v)) + cell[0]
        y = radius * np.outer(np.sin(u), np.sin(v)) + cell[1]
        z = radius * np.outer(np.ones(np.size(u)), np.cos(v)) + cell[2]
        ax.plot_surface(x, y, z, color=c, alpha=alpha, **kwargs)
        if labels:
            ax.text(
                cell[0],
                cell[1],
                cell[2],
                str(i),
                horizontalalignment="center",
                verticalalignment="center",
            )

    # Show colorbar
    if colorbar:
        sm = plt.cm.ScalarMappable(cmap=cm, norm=plt.Normalize(vmin=0, vmax=max_val))
        sm._A = []
        cbar = plt.colorbar(sm, ax=ax, fraction=0.05, alpha=alpha)
        cbar.set_label(f"{color_field.capitalize()} {color_index}", labelpad=20)

    # Calculate ax limits
    xmin, ymin, zmin = np.min(state.position[alive_cells], axis=0)
    xmax, ymax, zmax = np.max(state.position[alive_cells], axis=0)

    max_coord = max([xmax, ymax, zmax]) + 1
    min_coord = min([xmin, ymin, zmin]) - 1

    ax.set_xlim(min_coord, max_coord)
    ax.set_ylim(min_coord, max_coord)
    ax.set_zlim(min_coord, max_coord)

    # Set aspect ratio
    ax.set_box_aspect([1, 1, 1])  # Aspect ratio is 1:1:1

    # Set background color for ax
    ax.set_facecolor([1, 1, 1])

    if grid:
        ax.grid(alpha=0.2)
    else:
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

        # Make the box invisible
        ax.xaxis.pane.set_visible(False)
        ax.yaxis.pane.set_visible(False)
        ax.zaxis.pane.set_visible(False)

        # Hide the axes lines
        ax.xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))

    background_color = [56 / 256] * 3
    fig.patch.set_facecolor(background_color)
    fig.patch.set_alpha(0)

    fig.set_size_inches(8, 8)

    ax.view_init(elev=elev, azim=azim)

    return fig, ax


def draw_spheres_division(
    state,
    probability=False,
    colorbar=True,
    ax=None,
    cm=plt.cm.coolwarm,
    grid=False,
    labels=False,
    edges=False,
    cm_edges=plt.cm.coolwarm,
    elev=70,
    azim=-80,
    **kwargs,
):

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = plt.gcf()

    if "alpha" in kwargs:
        alpha = kwargs["alpha"]
        kwargs.pop("alpha")
    else:
        alpha = 0.6

    alive_cells = np.squeeze(state.celltype.sum(1) > 0)
    division = state.division[alive_cells]
    division = (division - division.min() + 1e-20) / (
        division.max() - division.min() + 1e-20
    )
    color = cm(division)

    if edges:
        ct_color = cm_edges(np.float32(state.celltype - 1)[alive_cells])
        for i, (cell, radius, c, ctc) in enumerate(
            zip(
                state.position[alive_cells],
                state.radius[alive_cells].squeeze(),
                color,
                ct_color,
            )
        ):
            u = np.linspace(0, 2 * np.pi, 100)
            v = np.linspace(0, np.pi, 100)
            x = radius * np.outer(np.cos(u), np.sin(v)) + cell[0]
            y = radius * np.outer(np.sin(u), np.sin(v)) + cell[1]
            z = radius * np.outer(np.ones(np.size(u)), np.cos(v)) + cell[2]
            ax.plot_surface(
                x, y, z, color=c, edgecolors=ctc, linewidth=0.5, alpha=alpha, **kwargs
            )
            if labels:
                ax.text(
                    cell[0],
                    cell[1],
                    cell[2],
                    str(i),
                    horizontalalignment="center",
                    verticalalignment="center",
                )
    else:
        for i, (cell, radius, c) in enumerate(
            zip(state.position[alive_cells], state.radius[alive_cells].squeeze(), color)
        ):
            u = np.linspace(0, 2 * np.pi, 100)
            v = np.linspace(0, np.pi, 100)
            x = radius * np.outer(np.cos(u), np.sin(v)) + cell[0]
            y = radius * np.outer(np.sin(u), np.sin(v)) + cell[1]
            z = radius * np.outer(np.ones(np.size(u)), np.cos(v)) + cell[2]
            ax.plot_surface(x, y, z, color=c, alpha=alpha, **kwargs)
            if labels:
                ax.text(
                    cell[0],
                    cell[1],
                    cell[2],
                    str(i),
                    horizontalalignment="center",
                    verticalalignment="center",
                )

    if colorbar:
        if probability:
            division = state.division[alive_cells] / (
                state.division[alive_cells].sum() + 1e-20
            )
            sm = plt.cm.ScalarMappable(
                cmap=cm, norm=plt.Normalize(vmin=division.min(), vmax=division.max())
            )
            sm._A = []
            cbar_text = "Division Probability"
        else:
            sm = plt.cm.ScalarMappable(
                cmap=cm,
                norm=plt.Normalize(
                    vmin=state.division[alive_cells].min(),
                    vmax=state.division[alive_cells].max(),
                ),
            )
            sm._A = []
            cbar_text = "Division Propensity"
        cbar = plt.colorbar(sm, ax=ax, fraction=0.05, alpha=0.5)
        cbar.set_label(cbar_text, labelpad=20)

    xmin, ymin, zmin = np.min(state.position[alive_cells], axis=0)
    xmax, ymax, zmax = np.max(state.position[alive_cells], axis=0)
    max_coord = max([xmax, ymax, zmax]) + 1
    min_coord = min([xmin, ymin, zmin]) - 1

    ax.set_xlim(min_coord, max_coord)
    ax.set_ylim(min_coord, max_coord)
    ax.set_zlim(min_coord, max_coord)

    ax.set_box_aspect([1, 1, 1])
    ax.set_facecolor([1, 1, 1])

    if grid:
        ax.grid(alpha=0.2)
    else:
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

        # Make the box invisible
        ax.xaxis.pane.set_visible(False)
        ax.yaxis.pane.set_visible(False)
        ax.zaxis.pane.set_visible(False)

        # Hide the axes lines
        ax.xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))

    background_color = [56 / 256] * 3
    fig.patch.set_facecolor(background_color)
    fig.patch.set_alpha(0)

    fig.set_size_inches(8, 8)
    ax.view_init(elev=elev, azim=azim)

    return fig, ax
