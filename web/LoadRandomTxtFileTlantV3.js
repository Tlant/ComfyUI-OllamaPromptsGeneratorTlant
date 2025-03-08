import { app } from "../../scripts/app.js";  

// 使用直接调试输出确认脚本加载  
console.log("LoadRandomTxtFileTlantV3 extension loading...");  

app.registerExtension({  
    name: "LoadRandomTxtFileTlantV3Extension",  
    async beforeRegisterNodeDef(nodeType, nodeData) {  
        // 输出节点名称以便调试  
        console.log("Checking node:", nodeData.name);  
        
        // 确保只处理我们的节点  
        if (nodeData.name !== "LoadRandomTxtFileTlantV3") {  
            return;  
        }  
        
        console.log("LoadRandomTxtFileTlantV3 node found, extending...");  
        
        // 修改原型  
        const origOnNodeCreated = nodeType.prototype.onNodeCreated;  
        nodeType.prototype.onNodeCreated = function() {  
            const result = origOnNodeCreated ? origOnNodeCreated.apply(this, arguments) : undefined;  
            
            console.log("Creating Load button for node:", this.id);  
            
            // 添加按钮，确保使用正确的方法  
            this.addWidget("button", "Load", null, () => {  
                const dirPathWidget = this.widgets.find(w => w.name === "dir_path");  
                if (!dirPathWidget || !dirPathWidget.value.trim()) {  
                    alert("Please enter a directory path first");  
                    return;  
                }  
                
                console.log("Load button clicked, path:", dirPathWidget.value);  
                this.loadSubDirectories(dirPathWidget.value);  
            });  
            
            // 保存引用  
            const self = this;  
            
            // 设置子目录变更逻辑  
            const subDirListWidget = this.widgets.find(w => w.name === "sub_dir_list");  
            if (subDirListWidget) {  
                const originalCallback = subDirListWidget.callback;  
                
                subDirListWidget.callback = function(value) {  
                    if (originalCallback) {  
                        originalCallback.call(this, value);  
                    }  
                    
                    const dirPathWidget = self.widgets.find(w => w.name === "dir_path");  
                    const selectedSubDirWidget = self.widgets.find(w => w.name === "selected_sub_dir");  
                    
                    if (dirPathWidget && selectedSubDirWidget) {  
                        const basePath = dirPathWidget.value;  
                        let newPath = basePath;  
                        
                        if (value && value !== "") {  
                            newPath = basePath.endsWith("/") || basePath.endsWith("\\")   
                                ? `${basePath}${value}`   
                                : `${basePath}/${value}`;  
                        }  
                        
                        selectedSubDirWidget.value = newPath;  
                        self.graph.setDirtyCanvas(true);  
                    }  
                };  
            }  
            
            return result;  
        };  
        
        // 添加加载子目录的方法  
        nodeType.prototype.loadSubDirectories = async function(dirPath) {  
            console.log("Loading subdirectories for:", dirPath);  
            
            try {  
                const response = await fetch("/get_subdirectories", {  
                    method: "POST",  
                    headers: { "Content-Type": "application/json" },  
                    body: JSON.stringify({ dir_path: dirPath })  
                });  
                
                if (!response.ok) {  
                    throw new Error(`Server responded with ${response.status}`);  
                }  
                
                const data = await response.json();  
                console.log("Received subdirectories:", data.subdirs);  
                
                const subDirListWidget = this.widgets.find(w => w.name === "sub_dir_list");  
                if (subDirListWidget) {  
                    subDirListWidget.options.values = data.subdirs || [""];  
                    subDirListWidget.value = "";  
                    
                    if (typeof subDirListWidget.callback === "function") {  
                        subDirListWidget.callback(subDirListWidget.value);  
                    }  
                    
                    this.graph.setDirtyCanvas(true);  
                }  
            } catch (error) {  
                console.error("Error loading subdirectories:", error);  
                alert(`Failed to load subdirectories: ${error.message}`);  
            }  
        };  
    }  
});  