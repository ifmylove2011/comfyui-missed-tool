const { app } = window.comfyAPI.app;

const NODE_NAME = "MissedXYLoraAxis";
const VERSION_PROPERTY = "missed_xy_lora_dynamic_version";
const CURRENT_VERSION = 1;
const LEGACY_PICKER_COUNT = 8;
const MIN_PICKERS = 1;
const MAX_PICKERS = 100;

function pickerIndex(name) {
    const match = /^lora_(\d+)$/.exec(name ?? "");
    return match ? Number(match[1]) : null;
}

function setWidgetVisible(widget, visible) {
    if (!widget) return;
    if (widget._missedOriginalType === undefined) {
        widget._missedOriginalType = widget.type;
        widget._missedOriginalComputeSize = widget.computeSize;
    }
    widget.type = visible ? widget._missedOriginalType : "missed-hidden-widget";
    widget.computeSize = visible
        ? widget._missedOriginalComputeSize
        : () => [0, -4];
}

function normalizeCount(value) {
    const parsed = Math.round(Number(value));
    if (!Number.isFinite(parsed)) return 4;
    return Math.max(MIN_PICKERS, Math.min(MAX_PICKERS, parsed));
}

app.registerExtension({
    name: "missed-tool.dynamic-lora-axis",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_NAME) return;

        const loraSpec = nodeData.input?.required?.lora_1;
        const loraValues = Array.isArray(loraSpec?.[0]) ? loraSpec[0] : ["None"];

        const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = originalOnNodeCreated?.apply(this, arguments);
            const node = this;

            node._missedRebuildLoraPickers = (requestedCount) => {
                const count = normalizeCount(requestedCount);
                const countWidget = node.widgets?.find((widget) => widget.name === "lora_count");
                if (countWidget) countWidget.value = count;

                for (let index = 2; index <= count; index++) {
                    const name = `lora_${index}`;
                    if (!node.widgets?.some((widget) => widget.name === name)) {
                        const widget = node.addWidget(
                            "combo",
                            name,
                            "None",
                            null,
                            { values: loraValues }
                        );
                        widget.inputSpec = loraSpec;

                        // Keep every LoRA selector together and the Update button last.
                        const buttonIndex = node.widgets.indexOf(node._missedLoraUpdateButton);
                        if (buttonIndex >= 0) {
                            node.widgets.splice(node.widgets.indexOf(widget), 1);
                            node.widgets.splice(buttonIndex, 0, widget);
                        }
                    }
                }

                for (const widget of node.widgets ?? []) {
                    const index = pickerIndex(widget.name);
                    if (index !== null) setWidgetVisible(widget, index <= count);
                }

                const computed = node.computeSize();
                node.setSize([Math.max(node.size[0], computed[0]), computed[1]]);
                node.graph?.setDirtyCanvas(true, true);
            };

            node._missedRebuildLoraPickers(
                node.widgets?.find((widget) => widget.name === "lora_count")?.value
            );
            node._missedLoraUpdateButton = node.addWidget(
                "button",
                "Update LoRA slots",
                null,
                () => node._missedRebuildLoraPickers(
                    node.widgets?.find((widget) => widget.name === "lora_count")?.value
                ),
                { serialize: false }
            );
            // LiteGraph checks the widget property itself when serializing;
            // the option alone is not sufficient on every frontend version.
            node._missedLoraUpdateButton.serialize = false;
            return result;
        };

        const originalOnConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (info) {
            let configuredInfo = info;
            const values = info?.widgets_values;
            const isLegacy = Array.isArray(values)
                && info?.properties?.[VERSION_PROPERTY] !== CURRENT_VERSION
                && typeof values[3] !== "number";

            if (isLegacy) {
                configuredInfo = {
                    ...info,
                    properties: { ...(info.properties ?? {}), [VERSION_PROPERTY]: CURRENT_VERSION },
                    widgets_values: [
                        ...values.slice(0, 3),
                        LEGACY_PICKER_COUNT,
                        ...values.slice(3, 3 + LEGACY_PICKER_COUNT),
                    ],
                };
            }

            const configuredValues = configuredInfo?.widgets_values;
            const targetCount = Array.isArray(configuredValues)
                ? normalizeCount(configuredValues[3])
                : normalizeCount(
                    this.widgets?.find((widget) => widget.name === "lora_count")?.value
                );
            // Hidden pickers are still serialized. Recreate all of them before
            // LiteGraph restores widget values, then hide those above the
            // requested count again after configuration.
            const namedPickerIndices = Object.keys(
                configuredInfo?.widgets_values_named ?? {}
            )
                .map(pickerIndex)
                .filter((index) => index !== null);
            const namedPickerCount = namedPickerIndices.length
                ? Math.max(...namedPickerIndices)
                : 0;
            const serializedPickerCount = namedPickerCount || (
                Array.isArray(configuredValues) ? configuredValues.length - 4 : 0
            );
            const storedPickerCount = Math.max(targetCount, serializedPickerCount);
            this._missedRebuildLoraPickers?.(storedPickerCount);
            const countWidget = this.widgets?.find((widget) => widget.name === "lora_count");
            if (countWidget) countWidget.value = targetCount;
            const result = originalOnConfigure?.call(this, configuredInfo);
            this._missedRebuildLoraPickers?.(targetCount);
            return result;
        };

        const originalOnSerialize = nodeType.prototype.onSerialize;
        nodeType.prototype.onSerialize = function (info) {
            const result = originalOnSerialize?.apply(this, arguments);
            info.properties ??= {};
            info.properties[VERSION_PROPERTY] = CURRENT_VERSION;
            return result;
        };
    },
});
