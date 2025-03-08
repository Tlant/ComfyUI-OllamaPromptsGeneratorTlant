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
        
        // 保存原始onDrawForeground函数引用  
        const origOnDrawForeground = nodeType.prototype.onDrawForeground;  
        
        // 修改原型  
        const origOnNodeCreated = nodeType.prototype.onNodeCreated;  
        nodeType.prototype.onNodeCreated = function() {  
            const result = origOnNodeCreated ? origOnNodeCreated.apply(this, arguments) : undefined;  
            
            console.log("Creating Load button for node:", this.id);  
            
            // 保存this引用  
            const self = this;  
            
            // 为节点添加一个subdirs属性来存储可用的子目录  
            this.subdirs = [""];  
            
            // 找到原始的sub_dir_list小部件并记住它的位置  
            const subDirListWidget = this.widgets.find(w => w.name === "sub_dir_list");  
            const subDirListIndex = subDirListWidget ? this.widgets.indexOf(subDirListWidget) : -1;  
            
            // 如果找到了，删除它  
            if (subDirListIndex !== -1) {  
                this.widgets.splice(subDirListIndex, 1);  
            }  
            
            // 创建一个自定义的combo widget  
            const comboWidget = this.addWidget("combo", "sub_dir_list", "", function(value) {  
                // 当选择变化时更新路径  
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
                    
                    console.log(`Updating selected_sub_dir to: ${newPath}`);  
                    selectedSubDirWidget.value = newPath;  
                    self.setDirtyCanvas(true);  
                }  
            }, { values: this.subdirs });  
            
            // 移动combo widget到原始位置  
            if (subDirListIndex !== -1) {  
                const currentIndex = this.widgets.indexOf(comboWidget);  
                this.widgets.splice(currentIndex, 1);  
                this.widgets.splice(subDirListIndex, 0, comboWidget);  
            }  
            
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
                
                // 保存子目录列表  
                this.subdirs = data.subdirs || [""];  
                
                // 找到我们的combo widget并更新它的值  
                const comboWidget = this.widgets.find(w => w.name === "sub_dir_list");  
                if (comboWidget) {  
                    comboWidget.options.values = this.subdirs;  
                    comboWidget.value = "";  
                    
                    // 触发更新selected_sub_dir  
                    if (typeof comboWidget.callback === "function") {  
                        comboWidget.callback(comboWidget.value);  
                    }  
                    
                    this.setDirtyCanvas(true);  
                }  
            } catch (error) {  
                console.error("Error loading subdirectories:", error);  
                alert(`Failed to load subdirectories: ${error.message}`);  
            }  
        };  

        // 重载onDrawForeground以确保combo widget正确渲染  
        nodeType.prototype.onDrawForeground = function(ctx) {  
            if (origOnDrawForeground) {  
                origOnDrawForeground.apply(this, arguments);  
            }  
            
            // 确保combo widget的当前值在列表中  
            const comboWidget = this.widgets.find(w => w.name === "sub_dir_list");  
            if (comboWidget && !comboWidget.options.values.includes(comboWidget.value)) {  
                // 如果当前值不在列表中但有效，添加到列表中  
                if (comboWidget.value !== undefined && comboWidget.value !== null) {  
                    comboWidget.options.values = [...this.subdirs];  
                    if (!comboWidget.options.values.includes(comboWidget.value)) {  
                        comboWidget.options.values.push(comboWidget.value);  
                    }  
                }  
            }  
        };  
    }  
});  