import { app, ComfyApp, ANIM_PREVIEW_WIDGET } from "../../scripts/app.js";
import { $el } from "../../scripts/ui.js";
import { api } from "../../scripts/api.js";
import { createImageHost, calculateImageGrid } from "../../scripts/ui/imagePreview.js";

function patchLiteGraph() {
    var temp_vec2 = new Float32Array(2);

    LGraphCanvas.prototype.drawNode = function(node, ctx) {
        var glow = false;
        this.current_node = node;

        var color = node.color || node.constructor.color || LiteGraph.NODE_DEFAULT_COLOR;
        var bgcolor = node.bgcolor || node.constructor.bgcolor || LiteGraph.NODE_DEFAULT_BGCOLOR;

        //shadow and glow
        if (node.mouseOver) {
            glow = true;
        }

        var low_quality = this.ds.scale < 0.6; //zoomed out

        //only render if it forces it to do it
        if (this.live_mode) {
            if (!node.flags.collapsed) {
                ctx.shadowColor = "transparent";
                if (node.onDrawForeground) {
                    node.onDrawForeground(ctx, this, this.canvas);
                }
            }
            return;
        }

        var editor_alpha = this.editor_alpha;
        ctx.globalAlpha = editor_alpha;

        if (this.render_shadows && !low_quality) {
            ctx.shadowColor = LiteGraph.DEFAULT_SHADOW_COLOR;
            ctx.shadowOffsetX = 2 * this.ds.scale;
            ctx.shadowOffsetY = 2 * this.ds.scale;
            ctx.shadowBlur = 3 * this.ds.scale;
        } else {
            ctx.shadowColor = "transparent";
        }

        //custom draw collapsed method (draw after shadows because they are affected)
        if (
            node.flags.collapsed &&
            node.onDrawCollapsed &&
            node.onDrawCollapsed(ctx, this) == true
        ) {
            return;
        }

        //clip if required (mask)
        var shape = node._shape || LiteGraph.BOX_SHAPE;
        var size = temp_vec2;
        temp_vec2.set(node.size);
        var horizontal = node.horizontal; // || node.flags.horizontal;

        if (node.flags.collapsed) {
            ctx.font = this.inner_text_font;
            var title = node.getTitle ? node.getTitle() : node.title;
            if (title != null) {
                node._collapsed_width = Math.min(
                    node.size[0],
                    ctx.measureText(title).width +
                        LiteGraph.NODE_TITLE_HEIGHT * 2
                ); //LiteGraph.NODE_COLLAPSED_WIDTH;
                size[0] = node._collapsed_width;
                size[1] = 0;
            }
        }

        if (node.clip_area) {
            //Start clipping
            ctx.save();
            ctx.beginPath();
            if (shape == LiteGraph.BOX_SHAPE) {
                ctx.rect(0, 0, size[0], size[1]);
            } else if (shape == LiteGraph.ROUND_SHAPE) {
                ctx.roundRect(0, 0, size[0], size[1], [10]);
            } else if (shape == LiteGraph.CIRCLE_SHAPE) {
                ctx.arc(
                    size[0] * 0.5,
                    size[1] * 0.5,
                    size[0] * 0.5,
                    0,
                    Math.PI * 2
                );
            }
            ctx.clip();
        }

        //draw shape
        if (node.has_errors) {
            bgcolor = "red";
        }
        this.drawNodeShape(
            node,
            ctx,
            size,
            color,
            bgcolor,
            node.is_selected,
            node.mouseOver
        );
        ctx.shadowColor = "transparent";

        //draw foreground
        if (node.onDrawForeground) {
            node.onDrawForeground(ctx, this, this.canvas);
        }

        //connection slots
        ctx.textAlign = horizontal ? "center" : "left";
        ctx.font = this.inner_text_font;

        var render_text = !low_quality;

        var out_slot = this.connecting_output;
        var in_slot = this.connecting_input;
        ctx.lineWidth = 1;

        var max_y = 0;
        var slot_pos = new Float32Array(2); //to reuse

        //render inputs and outputs
        if (!node.flags.collapsed) {
            //input connection slots
            if (node.inputs) {
                for (var i = 0; i < node.inputs.length; i++) {
                    var slot = node.inputs[i];

                    var slot_type = slot.type;
                    var slot_shape = slot.shape;

                    ctx.globalAlpha = editor_alpha;
                    //change opacity of incompatible slots when dragging a connection
                    if ( this.connecting_output && !LiteGraph.isValidConnection( slot.type , out_slot.type) ) {
                        ctx.globalAlpha = 0.4 * editor_alpha;
                    }

                    ctx.fillStyle =
                        slot.link != null
                            ? slot.color_on ||
                              this.default_connection_color_byType[slot_type] ||
                              this.default_connection_color.input_on
                            : slot.color_off ||
                              this.default_connection_color_byTypeOff[slot_type] ||
                              this.default_connection_color_byType[slot_type] ||
                              this.default_connection_color.input_off;

                    var pos = node.getConnectionPos(true, i, slot_pos);
                    pos[0] -= node.pos[0];
                    pos[1] -= node.pos[1];
                    if (max_y < pos[1] + LiteGraph.NODE_SLOT_HEIGHT * 0.5) {
                        max_y = pos[1] + LiteGraph.NODE_SLOT_HEIGHT * 0.5;
                    }

                    ctx.beginPath();

					if (slot_type == "array"){
                        slot_shape = LiteGraph.GRID_SHAPE; // place in addInput? addOutput instead?
                    }

                    var doStroke = true;

                    if (
                        slot.type === LiteGraph.EVENT ||
                        slot.shape === LiteGraph.BOX_SHAPE
                    ) {
                        if (horizontal) {
                            ctx.rect(
                                pos[0] - 5 + 0.5,
                                pos[1] - 8 + 0.5,
                                10,
                                14
                            );
                        } else {
                            ctx.rect(
                                pos[0] - 6 + 0.5,
                                pos[1] - 5 + 0.5,
                                14,
                                10
                            );
                        }
                    } else if (slot_shape === LiteGraph.ARROW_SHAPE) {
                        ctx.moveTo(pos[0] + 8, pos[1] + 0.5);
                        ctx.lineTo(pos[0] - 4, pos[1] + 6 + 0.5);
                        ctx.lineTo(pos[0] - 4, pos[1] - 6 + 0.5);
                        ctx.closePath();
                    } else if (slot_shape === LiteGraph.GRID_SHAPE) {
                        ctx.rect(pos[0] - 4, pos[1] - 4, 2, 2);
                        ctx.rect(pos[0] - 1, pos[1] - 4, 2, 2);
                        ctx.rect(pos[0] + 2, pos[1] - 4, 2, 2);
                        ctx.rect(pos[0] - 4, pos[1] - 1, 2, 2);
                        ctx.rect(pos[0] - 1, pos[1] - 1, 2, 2);
                        ctx.rect(pos[0] + 2, pos[1] - 1, 2, 2);
                        ctx.rect(pos[0] - 4, pos[1] + 2, 2, 2);
                        ctx.rect(pos[0] - 1, pos[1] + 2, 2, 2);
                        ctx.rect(pos[0] + 2, pos[1] + 2, 2, 2);
                        doStroke = false;
                    } else {
						if(low_quality)
	                        ctx.rect(pos[0] - 4, pos[1] - 4, 8, 8 ); //faster
						else
	                        ctx.arc(pos[0], pos[1], 4, 0, Math.PI * 2);
                    }
                    ctx.fill();

                    //render name
                    if (render_text) {
                        var text = slot.label != null ? slot.label : slot.name;
                        if (text) {
                            ctx.fillStyle = LiteGraph.NODE_TEXT_COLOR;
                            if (horizontal || slot.dir == LiteGraph.UP) {
                                ctx.fillText(text, pos[0], pos[1] - 10);
                            } else {
                                ctx.fillText(text, pos[0] + 10, pos[1] + 5);
                            }
                        }
                    }
                }
            }

            //output connection slots

            ctx.textAlign = horizontal ? "center" : "right";
            ctx.strokeStyle = "black";
            if (node.outputs) {
                for (var i = 0; i < node.outputs.length; i++) {
                    var slot = node.outputs[i];

                    var slot_type = slot.type;
                    var slot_shape = slot.shape;

                    //change opacity of incompatible slots when dragging a connection
                    if (this.connecting_input && !LiteGraph.isValidConnection( slot_type , in_slot.type) ) {
                        ctx.globalAlpha = 0.4 * editor_alpha;
                    }

                    var pos = node.getConnectionPos(false, i, slot_pos);
                    pos[0] -= node.pos[0];
                    pos[1] -= node.pos[1];
                    if (max_y < pos[1] + LiteGraph.NODE_SLOT_HEIGHT * 0.5) {
                        max_y = pos[1] + LiteGraph.NODE_SLOT_HEIGHT * 0.5;
                    }

                    ctx.fillStyle =
                        slot.links && slot.links.length
                            ? slot.color_on ||
                              this.default_connection_color_byType[slot_type] ||
                              this.default_connection_color.output_on
                            : slot.color_off ||
                              this.default_connection_color_byTypeOff[slot_type] ||
                              this.default_connection_color_byType[slot_type] ||
                              this.default_connection_color.output_off;
                    ctx.beginPath();
                    //ctx.rect( node.size[0] - 14,i*14,10,10);

					if (slot_type == "array"){
                        slot_shape = LiteGraph.GRID_SHAPE;
                    }

                    var doStroke = true;

                    if (
                        slot_type === LiteGraph.EVENT ||
                        slot_shape === LiteGraph.BOX_SHAPE
                    ) {
                        if (horizontal) {
                            ctx.rect(
                                pos[0] - 5 + 0.5,
                                pos[1] - 8 + 0.5,
                                10,
                                14
                            );
                        } else {
                            ctx.rect(
                                pos[0] - 6 + 0.5,
                                pos[1] - 5 + 0.5,
                                14,
                                10
                            );
                        }
                    } else if (slot_shape === LiteGraph.ARROW_SHAPE) {
                        ctx.moveTo(pos[0] + 8, pos[1] + 0.5);
                        ctx.lineTo(pos[0] - 4, pos[1] + 6 + 0.5);
                        ctx.lineTo(pos[0] - 4, pos[1] - 6 + 0.5);
                        ctx.closePath();
                    }  else if (slot_shape === LiteGraph.GRID_SHAPE) {
                        ctx.rect(pos[0] - 4, pos[1] - 4, 2, 2);
                        ctx.rect(pos[0] - 1, pos[1] - 4, 2, 2);
                        ctx.rect(pos[0] + 2, pos[1] - 4, 2, 2);
                        ctx.rect(pos[0] - 4, pos[1] - 1, 2, 2);
                        ctx.rect(pos[0] - 1, pos[1] - 1, 2, 2);
                        ctx.rect(pos[0] + 2, pos[1] - 1, 2, 2);
                        ctx.rect(pos[0] - 4, pos[1] + 2, 2, 2);
                        ctx.rect(pos[0] - 1, pos[1] + 2, 2, 2);
                        ctx.rect(pos[0] + 2, pos[1] + 2, 2, 2);
                        doStroke = false;
                    } else {
						if(low_quality)
	                        ctx.rect(pos[0] - 4, pos[1] - 4, 8, 8 );
						else
	                        ctx.arc(pos[0], pos[1], 4, 0, Math.PI * 2);
                    }

                    //trigger
                    //if(slot.node_id != null && slot.slot == -1)
                    //	ctx.fillStyle = "#F85";

                    //if(slot.links != null && slot.links.length)
                    ctx.fill();
					// if(!low_quality && doStroke)
	    //                 ctx.stroke();

                    //render output name
                    if (render_text) {
                        var text = slot.label != null ? slot.label : slot.name;
                        if (text) {
                            ctx.fillStyle = LiteGraph.NODE_TEXT_COLOR;
                            if (horizontal || slot.dir == LiteGraph.DOWN) {
                                ctx.fillText(text, pos[0], pos[1] - 8);
                            } else {
                                ctx.fillText(text, pos[0] - 10, pos[1] + 5);
                            }
                        }
                    }
                }
            }

            ctx.textAlign = "left";
            ctx.globalAlpha = 1;

            if (node.widgets) {
				var widgets_y = max_y;
                if (horizontal || node.widgets_up) {
                    widgets_y = 2;
                }
				if( node.widgets_start_y != null )
                    widgets_y = node.widgets_start_y;
                this.drawNodeWidgets(
                    node,
                    widgets_y,
                    ctx,
                    this.node_widget && this.node_widget[0] == node
                        ? this.node_widget[1]
                        : null
                );
            }
        } else if (this.render_collapsed_slots) {
            //if collapsed
            var input_slot = null;
            var output_slot = null;

            //get first connected slot to render
            if (node.inputs) {
                for (var i = 0; i < node.inputs.length; i++) {
                    var slot = node.inputs[i];
                    if (slot.link == null) {
                        continue;
                    }
                    input_slot = slot;
                    break;
                }
            }
            if (node.outputs) {
                for (var i = 0; i < node.outputs.length; i++) {
                    var slot = node.outputs[i];
                    if (!slot.links || !slot.links.length) {
                        continue;
                    }
                    output_slot = slot;
                }
            }

            if (input_slot) {
                var x = 0;
                var y = LiteGraph.NODE_TITLE_HEIGHT * -0.5; //center
                if (horizontal) {
                    x = node._collapsed_width * 0.5;
                    y = -LiteGraph.NODE_TITLE_HEIGHT;
                }
                ctx.fillStyle = "#686";
                ctx.beginPath();
                if (
                    slot.type === LiteGraph.EVENT ||
                    slot.shape === LiteGraph.BOX_SHAPE
                ) {
                    ctx.rect(x - 7 + 0.5, y - 4, 14, 8);
                } else if (slot.shape === LiteGraph.ARROW_SHAPE) {
                    ctx.moveTo(x + 8, y);
                    ctx.lineTo(x + -4, y - 4);
                    ctx.lineTo(x + -4, y + 4);
                    ctx.closePath();
                } else {
                    ctx.arc(x, y, 4, 0, Math.PI * 2);
                }
                ctx.fill();
            }

            if (output_slot) {
                var x = node._collapsed_width;
                var y = LiteGraph.NODE_TITLE_HEIGHT * -0.5; //center
                if (horizontal) {
                    x = node._collapsed_width * 0.5;
                    y = 0;
                }
                ctx.fillStyle = "#686";
                ctx.strokeStyle = "black";
                ctx.beginPath();
                if (
                    slot.type === LiteGraph.EVENT ||
                    slot.shape === LiteGraph.BOX_SHAPE
                ) {
                    ctx.rect(x - 7 + 0.5, y - 4, 14, 8);
                } else if (slot.shape === LiteGraph.ARROW_SHAPE) {
                    ctx.moveTo(x + 6, y);
                    ctx.lineTo(x - 6, y - 4);
                    ctx.lineTo(x - 6, y + 4);
                    ctx.closePath();
                } else {
                    ctx.arc(x, y, 4, 0, Math.PI * 2);
                }
                ctx.fill();
                //ctx.stroke();
            }
        }

        if (node.clip_area) {
            ctx.restore();
        }

        ctx.globalAlpha = 1.0;
    }
}

function patchApp() {
    app._invokeExtensionsAsync = async function(method, ...args) {
        return await Promise.all(
            this.extensions.map(async (ext) => {
                if (method in ext) {
                    try {
                        return await ext[method](...args, this);
                    } catch (error) {
                        console.error(
                            `Error calling extension '${ext.name}' method '${method}'`,
                            { error },
                            { extension: ext },
                            { args }
                        );
                    }
                }
            })
        );
    }

    app._addNodeContextMenuHandler = function(node) {
		function getCopyImageOption(img) {
			if (typeof window.ClipboardItem === "undefined") return [];
			return [
				{
					content: "Copy Image",
					callback: async () => {
						const url = new URL(img.src);
						url.searchParams.delete("preview");

						const writeImage = async (blob) => {
							await navigator.clipboard.write([
								new ClipboardItem({
									[blob.type]: blob,
								}),
							]);
						};

						try {
							const data = await fetch(url);
							const blob = await data.blob();
							try {
								await writeImage(blob);
							} catch (error) {
								// Chrome seems to only support PNG on write, convert and try again
								if (blob.type !== "image/png") {
									const canvas = $el("canvas", {
										width: img.naturalWidth,
										height: img.naturalHeight,
									});
									const ctx = canvas.getContext("2d");
									let image;
									if (typeof window.createImageBitmap === "undefined") {
										image = new Image();
										const p = new Promise((resolve, reject) => {
											image.onload = resolve;
											image.onerror = reject;
										}).finally(() => {
											URL.revokeObjectURL(image.src);
										});
										image.src = URL.createObjectURL(blob);
										await p;
									} else {
										image = await createImageBitmap(blob);
									}
									try {
										ctx.drawImage(image, 0, 0);
										canvas.toBlob(writeImage, "image/png");
									} finally {
										if (typeof image.close === "function") {
											image.close();
										}
									}

									return;
								}
								throw error;
							}
						} catch (error) {
							alert("Error copying image: " + (error.message ?? error));
						}
					},
				},
			];
		}

		node.prototype.getExtraMenuOptions = function (_, options) {
			if (this.imgs) {
				// If this node has images then we add an open in new tab item
				let img;
				if (this.imageIndex != null) {
					// An image is selected so select that
					img = this.imgs[this.imageIndex];
				} else if (this.overIndex != null) {
					// No image is selected but one is hovered
					img = this.imgs[this.overIndex];
				}
				if (img) {
					options.unshift(
						{
							content: "Open Image",
							callback: () => {
								let url = new URL(img.src);
								url.searchParams.delete("preview");
								window.open(url, "_blank");
							},
						},
						...getCopyImageOption(img),
						{
							content: "Save Image",
							callback: () => {
								const a = document.createElement("a");
								let url = new URL(img.src);
								url.searchParams.delete("preview");
								a.href = url;
								a.setAttribute("download", new URLSearchParams(url.search).get("filename"));
								document.body.append(a);
								a.click();
								requestAnimationFrame(() => a.remove());
							},
						}
					);
				}
			}

			options.push({
				content: "Bypass",
				callback: (obj) => {
					if (this.mode === 4) this.mode = 0;
					else this.mode = 4;
					this.graph.change();
				},
			});

			// prevent conflict of clipspace content
			if (!ComfyApp.clipspace_return_node) {
				options.push({
					content: "Copy (Clipspace)",
					callback: (obj) => {
						ComfyApp.copyToClipspace(this);
					},
				});

				if (ComfyApp.clipspace != null) {
					options.push({
						content: "Paste (Clipspace)",
						callback: () => {
							ComfyApp.pasteFromClipspace(this);
						},
					});
				}

				if (ComfyApp.isImageNode(this)) {
					options.push({
						content: "Open in MaskEditor",
						callback: (obj) => {
							ComfyApp.copyToClipspace(this);
							ComfyApp.clipspace_return_node = this;
							ComfyApp.open_maskeditor();
						},
					});
				}
			}
		};
    }

    app._addDrawBackgroundHandler = function(node) {
		const app = this;

		function getImageTop(node) {
			let shiftY;
			if (node.imageOffset != null) {
				shiftY = node.imageOffset;
			} else {
				if (node.widgets?.length) {
					const w = node.widgets[node.widgets.length - 1];
					shiftY = w.last_y;
					if (w.computeSize) {
						shiftY += w.computeSize()[1] + 4;
					}
					else if(w.computedHeight) {
						shiftY += w.computedHeight;
					}
					else {
						shiftY += LiteGraph.NODE_WIDGET_HEIGHT + 4;
					}
				} else {
					shiftY = node.computeSize()[1];
				}
			}
			return shiftY;
		}

		node.prototype.setSizeForImage = function (force) {
			if(!force && this.animatedImages) return;

			if (this.inputHeight || this.freeWidgetSpace > 210) {
				this.setSize(this.size);
				return;
			}
			const minHeight = getImageTop(this) + 220;
			if (this.size[1] < minHeight) {
				this.setSize([this.size[0], minHeight]);
			}
		};

		node.prototype.onDrawBackground = function (ctx) {
			if (!this.flags.collapsed) {
				let imgURLs = []
				let imagesChanged = false

				const output = app.nodeOutputs[this.id + ""];
				if (output?.images) {
					this.animatedImages = output?.animated?.find(Boolean);
					if (this.images !== output.images) {
						this.images = output.images;
						imagesChanged = true;
						imgURLs = imgURLs.concat(
							output.images.map((params) => {
								return api.apiURL(
									"/view?" +
										new URLSearchParams(params).toString() +
										(this.animatedImages ? "" : app.getPreviewFormatParam()) + app.getRandParam()
								);
							})
						);
					}
				}

				const preview = app.nodePreviewImages[this.id + ""]
				if (this.preview !== preview) {
					this.preview = preview
					imagesChanged = true;
					if (preview != null) {
						imgURLs.push(preview);
					}
				}

				if (imagesChanged) {
					this.imageIndex = null;
					if (imgURLs.length > 0) {
						Promise.all(
							imgURLs.map((src) => {
								return new Promise((r) => {
									const img = new Image();
									img.onload = () => r(img);
									img.onerror = () => r(null);
									img.src = src
								});
							})
						).then((imgs) => {
							if ((!output || this.images === output.images) && (!preview || this.preview === preview)) {
								this.imgs = imgs.filter(Boolean);
								this.setSizeForImage?.();
								app.graph.setDirtyCanvas(true);
							}
						});
					}
					else {
						this.imgs = null;
					}
				}

				function calculateGrid(w, h, n) {
					let columns, rows, cellsize;

					if (w > h) {
						cellsize = h;
						columns = Math.ceil(w / cellsize);
						rows = Math.ceil(n / columns);
					} else {
						cellsize = w;
						rows = Math.ceil(h / cellsize);
						columns = Math.ceil(n / rows);
					}

					while (columns * rows < n) {
						cellsize++;
						if (w >= h) {
							columns = Math.ceil(w / cellsize);
							rows = Math.ceil(n / columns);
						} else {
							rows = Math.ceil(h / cellsize);
							columns = Math.ceil(n / rows);
						}
					}

					const cell_size = Math.min(w/columns, h/rows);
					return {cell_size, columns, rows};
				}

				function is_all_same_aspect_ratio(imgs) {
					// assume: imgs.length >= 2
					let ratio = imgs[0].naturalWidth/imgs[0].naturalHeight;

					for(let i=1; i<imgs.length; i++) {
						let this_ratio = imgs[i].naturalWidth/imgs[i].naturalHeight;
						if(ratio != this_ratio)
							return false;
					}

					return true;
				}

				if (this.imgs?.length) {
					const widgetIdx = this.widgets?.findIndex((w) => w.name === ANIM_PREVIEW_WIDGET);

					if(this.animatedImages) {
						// Instead of using the canvas we'll use a IMG
						if(widgetIdx > -1) {
							// Replace content
							const widget = this.widgets[widgetIdx];
							widget.options.host.updateImages(this.imgs);
						} else {
							const host = createImageHost(this);
							this.setSizeForImage(true);
							const widget = this.addDOMWidget(ANIM_PREVIEW_WIDGET, "img", host.el, {
								host,
								getHeight: host.getHeight,
								onDraw: host.onDraw,
								hideOnZoom: false
							});
							widget.serializeValue = () => undefined;
							widget.options.host.updateImages(this.imgs);
						}
						return;
					}

					if (widgetIdx > -1) {
						this.widgets[widgetIdx].onRemove?.();
						this.widgets.splice(widgetIdx, 1);
					}

					const canvas = app.graph.list_of_graphcanvas[0];
					const mouse = canvas.graph_mouse;
					if (!canvas.pointer_is_down && this.pointerDown) {
						if (mouse[0] === this.pointerDown.pos[0] && mouse[1] === this.pointerDown.pos[1]) {
							this.imageIndex = this.pointerDown.index;
						}
						this.pointerDown = null;
					}

					let imageIndex = this.imageIndex;
					const numImages = this.imgs.length;
					if (numImages === 1 && !imageIndex) {
						this.imageIndex = imageIndex = 0;
					}

					const top = getImageTop(this);
					var shiftY = top;

					let dw = this.size[0];
					let dh = this.size[1];
					dh -= shiftY;

					if (imageIndex == null) {
						var cellWidth, cellHeight, shiftX, cell_padding, cols;

						const compact_mode = is_all_same_aspect_ratio(this.imgs);
						if(!compact_mode) {
							// use rectangle cell style and border line
							cell_padding = 2;
							const { cell_size, columns, rows } = calculateGrid(dw, dh, numImages);
							cols = columns;

							cellWidth = cell_size;
							cellHeight = cell_size;
							shiftX = (dw-cell_size*cols)/2;
							shiftY = (dh-cell_size*rows)/2 + top;
						}
						else {
							cell_padding = 0;
							({ cellWidth, cellHeight, cols, shiftX } = calculateImageGrid(this.imgs, dw, dh));
						}

						let anyHovered = false;
						this.imageRects = [];
						for (let i = 0; i < numImages; i++) {
							const img = this.imgs[i];
							const row = Math.floor(i / cols);
							const col = i % cols;
							const x = col * cellWidth + shiftX;
							const y = row * cellHeight + shiftY;
							if (!anyHovered) {
								anyHovered = LiteGraph.isInsideRectangle(
									mouse[0],
									mouse[1],
									x + this.pos[0],
									y + this.pos[1],
									cellWidth,
									cellHeight
								);
								if (anyHovered) {
									this.overIndex = i;
									let value = 110;
									if (canvas.pointer_is_down) {
										if (!this.pointerDown || this.pointerDown.index !== i) {
											this.pointerDown = { index: i, pos: [...mouse] };
										}
										value = 125;
									}
									ctx.filter = `contrast(${value}%) brightness(${value}%)`;
									canvas.canvas.style.cursor = "pointer";
								}
							}
							this.imageRects.push([x, y, cellWidth, cellHeight]);

							let wratio = cellWidth/img.width;
							let hratio = cellHeight/img.height;
							var ratio = Math.min(wratio, hratio);

							let imgHeight = ratio * img.height;
							let imgY = row * cellHeight + shiftY + (cellHeight - imgHeight)/2;
							let imgWidth = ratio * img.width;
							let imgX = col * cellWidth + shiftX + (cellWidth - imgWidth)/2;

							ctx.drawImage(img, imgX+cell_padding, imgY+cell_padding, imgWidth-cell_padding*2, imgHeight-cell_padding*2);
							if(!compact_mode) {
								// rectangle cell and border line style
								ctx.strokeStyle = "#8F8F8F";
								ctx.lineWidth = 1;
								ctx.strokeRect(x+cell_padding, y+cell_padding, cellWidth-cell_padding*2, cellHeight-cell_padding*2);
							}

							ctx.filter = "none";
						}

						if (!anyHovered) {
							this.pointerDown = null;
							this.overIndex = null;
						}
					} else {
						// Draw individual
						let w = this.imgs[imageIndex].naturalWidth;
						let h = this.imgs[imageIndex].naturalHeight;

						const scaleX = dw / w;
						const scaleY = dh / h;
						const scale = Math.min(scaleX, scaleY, 1);

						w *= scale;
						h *= scale;

						let x = (dw - w) / 2;
						let y = (dh - h) / 2 + shiftY;
						ctx.drawImage(this.imgs[imageIndex], x, y, w, h);

						const drawButton = (x, y, sz, text) => {
							const hovered = LiteGraph.isInsideRectangle(mouse[0], mouse[1], x + this.pos[0], y + this.pos[1], sz, sz);
							let fill = "#333";
							let textFill = "#fff";
							let isClicking = false;
							if (hovered) {
								canvas.canvas.style.cursor = "pointer";
								if (canvas.pointer_is_down) {
									fill = "#1e90ff";
									isClicking = true;
								} else {
									fill = "#eee";
									textFill = "#000";
								}
							} else {
								this.pointerWasDown = null;
							}

							ctx.fillStyle = fill;
							ctx.beginPath();
							ctx.roundRect(x, y, sz, sz, [4]);
							ctx.fill();
							ctx.fillStyle = textFill;
							ctx.font = "12px Arial";
							ctx.textAlign = "center";
							ctx.fillText(text, x + 15, y + 20);

							return isClicking;
						};

						if (numImages > 1) {
							if (drawButton(dw - 40, dh + top - 40, 30, `${this.imageIndex + 1}/${numImages}`)) {
								let i = this.imageIndex + 1 >= numImages ? 0 : this.imageIndex + 1;
								if (!this.pointerDown || !this.pointerDown.index === i) {
									this.pointerDown = { index: i, pos: [...mouse] };
								}
							}

							if (drawButton(dw - 40, top + 10, 30, `x`)) {
								if (!this.pointerDown || !this.pointerDown.index === null) {
									this.pointerDown = { index: null, pos: [...mouse] };
								}
							}
						}
					}
				}
			}
		};
    }

	app._addNodeKeyHandler = function(node) {
		const app = this;
		const origNodeOnKeyDown = node.prototype.onKeyDown;

		node.prototype.onKeyDown = function(e) {
			if (origNodeOnKeyDown && origNodeOnKeyDown.apply(this, e) === false) {
				return false;
			}

			if (this.flags.collapsed || !this.imgs || this.imageIndex === null) {
				return;
			}

			let handled = false;

			if (e.key === "ArrowLeft" || e.key === "ArrowRight") {
				if (e.key === "ArrowLeft") {
					this.imageIndex -= 1;
				} else if (e.key === "ArrowRight") {
					this.imageIndex += 1;
				}
				this.imageIndex %= this.imgs.length;

				if (this.imageIndex < 0) {
					this.imageIndex = this.imgs.length + this.imageIndex;
				}
				handled = true;
			} else if (e.key === "Escape") {
				this.imageIndex = null;
				handled = true;
			}

			if (handled === true) {
				e.preventDefault();
				e.stopImmediatePropagation();
				return false;
			}
		}
    }

	app.registerNodeDef = async function(nodeId, nodeData) {
		const self = this;
		const node = Object.assign(
			function ComfyNode() {
				var inputs = nodeData["input"]["required"];
				if (nodeData["input"]["optional"] != undefined) {
					inputs = Object.assign({}, nodeData["input"]["required"], nodeData["input"]["optional"]);
				}
				const config = { minWidth: 1, minHeight: 1 };
				for (const inputName in inputs) {
					const inputData = inputs[inputName];
					const type = inputData[0];
					const extraInfo = {};

					let widgetCreated = true;
					const widgetType = self.getWidgetType(inputData, inputName);
					if(widgetType) {
						if(widgetType === "COMBO") {
							Object.assign(config, self.widgets.COMBO(this, inputName, inputData, app) || {});
						} else {
							Object.assign(config, self.widgets[widgetType](this, inputName, inputData, app) || {});
						}
					} else {
						// Node connection inputs
						if (inputData[1]?.multiple) {
							extraInfo.multiple = true;
							extraInfo.shape = LiteGraph.GRID_SHAPE;
						}
						this.addInput(inputName, type, extraInfo);
						widgetCreated = false;
					}

					if(widgetCreated && inputData[1]?.forceInput && config?.widget) {
						if (!config.widget.options) config.widget.options = {};
						config.widget.options.forceInput = inputData[1].forceInput;
					}
					if(widgetCreated && inputData[1]?.defaultInput && config?.widget) {
						if (!config.widget.options) config.widget.options = {};
						config.widget.options.defaultInput = inputData[1].defaultInput;
					}
				}

				for (const o in nodeData["output"]) {
					let output = nodeData["output"][o];
					if(output instanceof Array) output = "COMBO";
					const outputName = nodeData["output_name"][o] || output;
					const outputShape = nodeData["output_is_list"][o] ? LiteGraph.GRID_SHAPE : LiteGraph.CIRCLE_SHAPE ;
					this.addOutput(outputName, output, { shape: outputShape });
				}

				const s = this.computeSize();
				s[0] = Math.max(config.minWidth, s[0] * 1.5);
				s[1] = Math.max(config.minHeight, s[1]);
				this.size = s;
				this.serialize_widgets = true;

				app._invokeExtensionsAsync("nodeCreated", this);
			},
			{
				title: nodeData.display_name || nodeData.name,
				comfyClass: nodeData.name,
				nodeData
			}
		);
		node.prototype.comfyClass = nodeData.name;

		app._addNodeContextMenuHandler(node);
		app._addDrawBackgroundHandler(node, app);
		app._addNodeKeyHandler(node);

		await app._invokeExtensionsAsync("beforeRegisterNodeDef", node, nodeData);
		LiteGraph.registerNodeType(nodeId, node);
		node.category = nodeData.category;
    }
}

patchLiteGraph()
patchApp()
