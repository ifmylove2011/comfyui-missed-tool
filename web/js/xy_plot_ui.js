const { app } = window.comfyAPI.app;

const NODE_NAME = "MissedXYPlot";
const OPTIONAL_INPUTS = new Set([
    "clip",
    "x_axis",
    "y_axis",
    "custom_sampler",
    "custom_sigmas",
]);

function arrangeAndLabelOptionalInputs(node) {
    if (!Array.isArray(node.inputs)) return;

    for (const input of node.inputs) {
        if (OPTIONAL_INPUTS.has(input.name)) {
            input.label = `${input.name} (optional)`;
        }
    }

    const clipIndex = node.inputs.findIndex((input) => input.name === "clip");
    if (clipIndex >= 0) {
        const [clipInput] = node.inputs.splice(clipIndex, 1);
        const modelIndex = node.inputs.findIndex((input) => input.name === "model");
        node.inputs.splice(modelIndex >= 0 ? modelIndex + 1 : 0, 0, clipInput);
    }

    // Saved workflows store numeric target-slot indices. Recalculate them
    // after moving the slot so both old and newly saved workflows stay valid.
    for (const [index, input] of node.inputs.entries()) {
        if (input.link == null) continue;
        const link = node.graph?._links?.get?.(input.link)
            ?? node.graph?.links?.[input.link];
        if (link) link.target_slot = index;
    }

    node.graph?.setDirtyCanvas(true, true);
}

app.registerExtension({
    name: "missed-tool.xy-plot-ui",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_NAME) return;

        const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = originalOnNodeCreated?.apply(this, arguments);
            arrangeAndLabelOptionalInputs(this);
            return result;
        };

        const originalOnConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const result = originalOnConfigure?.apply(this, arguments);
            arrangeAndLabelOptionalInputs(this);
            return result;
        };
    },
});
