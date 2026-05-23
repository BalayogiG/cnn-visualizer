import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Play, Square, Settings, Database, Activity, RefreshCw, PenTool, Eraser, Eye, BrainCircuit } from 'lucide-react';
import * as tf from '@tensorflow/tfjs';

const CLASSES = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'];
const IMG_SIZE = 28;

export default function App() {
  const [status, setStatus] = useState('Ready. Step 1: Load MNIST Dataset.');
  const [modelStatus, setModelStatus] = useState('untrained'); // untrained, training, trained
  const [progress, setProgress] = useState(0);
  const [epochInfo, setEpochInfo] = useState('');
  
  const [model, setModel] = useState(null);
  const [activationModel, setActivationModel] = useState(null);
  const [dataset, setDataset] = useState(null);
  
  const [predictions, setPredictions] = useState(Array(10).fill(0));
  const [layerVis, setLayerVis] = useState({});
  const [hoveredNode, setHoveredNode] = useState(null);

  const canvasRef = useRef(null);
  const [isDrawing, setIsDrawing] = useState(false);
  const lastInferenceTime = useRef(0);

  useEffect(() => {
    if (canvasRef.current) {
      const canvas = canvasRef.current;
      const ctx = canvas.getContext('2d');
      ctx.fillStyle = 'black';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
    }
  }, []);

  const loadMnistDataset = async () => {
    setStatus('Downloading MNIST images...');
    setModelStatus('untrained');
    
    try {
      const img = new Image();
      img.crossOrigin = 'anonymous';
      img.src = 'https://storage.googleapis.com/learnjs-data/model-builder/mnist_images.png';
      await new Promise((resolve, reject) => { 
        img.onload = resolve; 
        img.onerror = () => reject(new Error('Failed to load MNIST images'));
      });

      setStatus('Downloading MNIST labels...');
      const labelsResponse = await fetch('https://storage.googleapis.com/learnjs-data/model-builder/mnist_labels_uint8');
      if (!labelsResponse.ok) throw new Error('Failed to load labels');
      const labelsBuffer = await labelsResponse.arrayBuffer();
      const labelsData = new Uint8Array(labelsBuffer);

      setStatus('Processing and shuffling data...');
      
      const numSamples = 5000; 
      const totalSamples = 65000;
      
      // Create an array of all indices and shuffle it to grab a random, balanced mix
      const indices = Array.from({ length: totalSamples }, (_, i) => i);
      for (let i = indices.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [indices[i], indices[j]] = [indices[j], indices[i]];
      }
      const selectedIndices = indices.slice(0, numSamples);
      
      const canvas = document.createElement('canvas');
      canvas.width = 784; 
      canvas.height = numSamples;
      const ctx = canvas.getContext('2d');
      const datasetYs = new Float32Array(numSamples * 10);

      // Extract the randomly selected rows and one-hot labels
      for (let i = 0; i < numSamples; i++) {
        const sourceY = selectedIndices[i];
        ctx.drawImage(img, 0, sourceY, 784, 1, 0, i, 784, 1);
        
        // MNIST labels are pre-one-hot encoded (10 bytes per image)
        for (let c = 0; c < 10; c++) {
          datasetYs[i * 10 + c] = labelsData[sourceY * 10 + c];
        }
      }

      const imgData = ctx.getImageData(0, 0, 784, numSamples);
      const flatInputs = new Float32Array(numSamples * 784);
      for (let i = 0; i < flatInputs.length; i++) {
        flatInputs[i] = imgData.data[i * 4] / 255.0; // Extract Red channel
      }

      if (dataset) {
        dataset.xs.dispose();
        dataset.ys.dispose();
      }

      tf.tidy(() => {
        const xs = tf.tensor4d(flatInputs, [numSamples, IMG_SIZE, IMG_SIZE, 1]);
        const ys = tf.tensor2d(datasetYs, [numSamples, 10]);
        setDataset({ xs: tf.keep(xs), ys: tf.keep(ys) });
      });

      setStatus(`MNIST ready (${numSamples} random samples). Step 2: Train Model.`);
    } catch (error) {
      console.error("Dataset Error:", error);
      setStatus('Error loading MNIST. Check console.');
    }
  };

  const buildAndTrainModel = async () => {
    if (!dataset) return;
    setStatus('Building model...');
    setModelStatus('training');
    setProgress(0);

    const newModel = tf.sequential();
    
    newModel.add(tf.layers.conv2d({
      inputShape: [IMG_SIZE, IMG_SIZE, 1],
      filters: 8, kernelSize: 3, activation: 'relu', padding: 'same', name: 'conv2d_1'
    }));
    newModel.add(tf.layers.maxPooling2d({ poolSize: 2, strides: 2, name: 'max_pool_1' }));
    
    newModel.add(tf.layers.conv2d({
      filters: 16, kernelSize: 3, activation: 'relu', padding: 'same', name: 'conv2d_2'
    }));
    newModel.add(tf.layers.maxPooling2d({ poolSize: 2, strides: 2, name: 'max_pool_2' }));
    
    newModel.add(tf.layers.flatten({ name: 'flatten' }));
    // Upgraded dense layer to 128 units so it can learn 10 digits effectively
    newModel.add(tf.layers.dense({ units: 128, activation: 'relu', name: 'dense_1' }));
    newModel.add(tf.layers.dense({ units: 10, activation: 'softmax', name: 'output' }));

    newModel.compile({
      optimizer: 'adam',
      loss: 'categoricalCrossentropy',
      metrics: ['accuracy'],
    });

    const layerNames = ['conv2d_1', 'max_pool_1', 'conv2d_2', 'max_pool_2', 'dense_1', 'output'];
    const outputs = layerNames.map(name => newModel.getLayer(name).output);
    const actModel = tf.model({ inputs: newModel.inputs, outputs: outputs });

    setModel(newModel);
    setActivationModel(actModel);

    setStatus('Training in progress...');

    const epochs = 20;
    try {
      await newModel.fit(dataset.xs, dataset.ys, {
        epochs: epochs,
        batchSize: 32,
        shuffle: true,
        callbacks: {
          onEpochEnd: async (epoch, logs) => {
            setProgress(((epoch + 1) / epochs) * 100);
            setEpochInfo(`Epoch ${epoch + 1}/${epochs} - Loss: ${logs.loss.toFixed(4)} - Acc: ${(logs.acc * 100).toFixed(1)}%`);
            await tf.nextFrame();
          }
        }
      });
      setStatus('Training Complete! Step 3: Draw below to test.');
      setModelStatus('trained');
    } catch (err) {
      console.error("Training Error:", err);
      setStatus('Error during training. Check console.');
      setModelStatus('untrained');
    }
  };

  const tensorToHeatmaps = (tensorData, shape) => {
    // Handle Flattened 1D Dense Layers
    if (shape.length === 2) {
      const units = shape[1];
      const maps = [];
      let w = 1, h = units;
      if (units === 64) { w = 8; h = 8; } // Fold 64 units into an 8x8 square
      else if (units === 128) { w = 16; h = 8; } // Fold 128 units into a 16x8 rectangle
      
      let min = Infinity, max = -Infinity;
      for (let i = 0; i < units; i++) {
        if (tensorData[i] < min) min = tensorData[i];
        if (tensorData[i] > max) max = tensorData[i];
      }
      const range = max - min === 0 ? 1 : max - min;
      
      const canvas = document.createElement('canvas');
      canvas.width = w;
      canvas.height = h;
      const ctx = canvas.getContext('2d');
      const imgData = ctx.createImageData(w, h);

      for (let i = 0; i < units; i++) {
        const val = tensorData[i];
        const norm = (val - min) / range;
        const color = Math.floor(norm * 255);
        imgData.data[i * 4] = 0;
        imgData.data[i * 4 + 1] = color; // Cyan tinted
        imgData.data[i * 4 + 2] = color;
        imgData.data[i * 4 + 3] = 255;
      }
      ctx.putImageData(imgData, 0, 0);
      maps.push({
        url: canvas.toDataURL(),
        min: min === Infinity ? "0" : min.toFixed(2),
        max: max === -Infinity ? "0" : max.toFixed(2),
        isDense: true
      });
      return maps;
    }

    const [batch, h, w, filters] = shape;
    const maps = [];
    const pixelsPerFilter = h * w;

    for (let f = 0; f < filters; f++) {
      let min = Infinity;
      let max = -Infinity;
      
      for (let i = 0; i < pixelsPerFilter; i++) {
        const val = tensorData[i * filters + f];
        if (val < min) min = val;
        if (val > max) max = val;
      }
      
      const range = max - min === 0 ? 1 : max - min;
      
      const canvas = document.createElement('canvas');
      canvas.width = w;
      canvas.height = h;
      const ctx = canvas.getContext('2d');
      const imgData = ctx.createImageData(w, h);

      for (let i = 0; i < pixelsPerFilter; i++) {
        const val = tensorData[i * filters + f];
        const norm = (val - min) / range;
        imgData.data[i * 4] = 0;
        imgData.data[i * 4 + 1] = Math.floor(norm * 255);
        imgData.data[i * 4 + 2] = Math.floor(norm * 255);
        imgData.data[i * 4 + 3] = 255;
      }
      
      ctx.putImageData(imgData, 0, 0);
      maps.push({
        url: canvas.toDataURL(),
        min: min === Infinity ? "0" : min.toFixed(2),
        max: max === -Infinity ? "0" : max.toFixed(2)
      });
    }
    return maps;
  };

  const runInference = useCallback((sourceCanvas) => {
    if (!model || !activationModel || modelStatus !== 'trained') return;

    try {
      tf.tidy(() => {
        const hiddenCanvas = document.createElement('canvas');
        hiddenCanvas.width = IMG_SIZE;
        hiddenCanvas.height = IMG_SIZE;
        const ctx = hiddenCanvas.getContext('2d');
        ctx.fillStyle = 'black';
        ctx.fillRect(0, 0, IMG_SIZE, IMG_SIZE);
        
        // MNIST digits are centered in a 20x20 box inside the 28x28 image
        // Adding padding simulates this, drastically improving accuracy!
        const padding = 4;
        const innerSize = IMG_SIZE - (padding * 2);
        ctx.drawImage(sourceCanvas, padding, padding, innerSize, innerSize);
        
        const imgData = ctx.getImageData(0, 0, IMG_SIZE, IMG_SIZE);
        
        const data = new Float32Array(IMG_SIZE * IMG_SIZE);
        let isBlank = true;
        for (let i = 0; i < data.length; i++) {
          const val = imgData.data[i * 4] / 255.0; 
          data[i] = val;
          if (val > 0.05) isBlank = false;
        }

        if (isBlank) {
          setPredictions(Array(10).fill(0));
          return;
        }

        const inputTensor = tf.tensor4d(data, [1, IMG_SIZE, IMG_SIZE, 1]);
        
        const predTensor = model.predict(inputTensor);
        const predData = predTensor.dataSync();
        setPredictions(Array.from(predData));

        const actTensors = activationModel.predict(inputTensor);
        const displayNames = ['Conv2D 1 (8 Filters)', 'MaxPool 1', 'Conv2D 2 (16 Filters)', 'MaxPool 2', 'Dense 1 (128 Units)', 'Output (10 Classes)'];
        const newLayerVis = {};

        newLayerVis['Model Input (28x28)'] = tensorToHeatmaps(data, [1, IMG_SIZE, IMG_SIZE, 1]);

        actTensors.forEach((act, idx) => {
          const dataSync = act.dataSync();
          const shape = act.shape;
          newLayerVis[displayNames[idx]] = tensorToHeatmaps(dataSync, shape);
        });

        setLayerVis(newLayerVis);
      });
    } catch (err) {
      console.error("Inference Error:", err);
    }
  }, [model, activationModel, modelStatus]);

  const getCoordinates = (e, canvas) => {
    const rect = canvas.getBoundingClientRect();
    const clientX = e.touches && e.touches.length > 0 ? e.touches[0].clientX : e.clientX;
    const clientY = e.touches && e.touches.length > 0 ? e.touches[0].clientY : e.clientY;
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    return { x: (clientX - rect.left) * scaleX, y: (clientY - rect.top) * scaleY };
  };

  const startDrawing = (e) => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const { x, y } = getCoordinates(e, canvas);
    ctx.strokeStyle = 'white';
    ctx.lineWidth = 24; // Optimized line thickness for MNIST
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.beginPath();
    ctx.moveTo(x, y);
    setIsDrawing(true);
  };

  const draw = (e) => {
    if (!isDrawing) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const { x, y } = getCoordinates(e, canvas);
    ctx.lineTo(x, y);
    ctx.stroke();
    
    // Throttle inference heavily to prevent UI lag
    const now = Date.now();
    if (now - lastInferenceTime.current > 60) {
      runInference(canvas);
      lastInferenceTime.current = now;
    }
  };

  const stopDrawing = () => {
    if (!isDrawing) return;
    setIsDrawing(false);
    runInference(canvasRef.current);
  };

  const clearCanvas = () => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = 'black';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    setPredictions(Array(10).fill(0));
    setLayerVis({});
  };

  return (
    <div className="min-h-screen bg-slate-900 text-slate-200 font-sans p-4 md:p-8 flex flex-col items-center">
      
      <header className="w-full max-w-6xl mb-8 flex flex-col md:flex-row items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <div className="p-3 bg-cyan-500/20 rounded-lg text-cyan-400">
            <BrainCircuit size={32} />
          </div>
          <div>
            <h1 className="text-2xl font-bold bg-gradient-to-r from-cyan-400 to-blue-500 bg-clip-text text-transparent">
              CNN Deep Vision
            </h1>
            <p className="text-slate-400 text-sm">Real-time Browser-based Neural Network Visualizer</p>
          </div>
        </div>

        <div className="flex items-center gap-3 bg-slate-800 px-4 py-2 rounded-full border border-slate-700 shadow-inner max-w-md w-full md:w-auto overflow-hidden">
          <Activity size={16} className={modelStatus === 'training' ? 'text-yellow-400 animate-pulse shrink-0' : 'text-slate-400 shrink-0'} />
          <span className="text-sm font-medium truncate">{status}</span>
        </div>
      </header>

      <div className="w-full max-w-6xl grid grid-cols-1 lg:grid-cols-4 gap-8">
        
        {/* Left Column Controls */}
        <div className="lg:col-span-1 flex flex-col gap-6">
          <div className="bg-slate-800 p-6 rounded-xl border border-slate-700 shadow-xl">
            <h3 className="text-lg font-semibold mb-4 flex items-center gap-2 border-b border-slate-700 pb-2">
              <Settings size={18} className="text-blue-400" />
              Controls
            </h3>
            
            <button 
              onClick={loadMnistDataset}
              disabled={modelStatus === 'training'}
              className="w-full mb-3 flex items-center justify-center gap-2 bg-slate-700 hover:bg-slate-600 disabled:opacity-50 disabled:cursor-not-allowed text-white py-2 px-4 rounded-lg transition-colors"
            >
              <Database size={16} />
              1. Load MNIST Data
            </button>
            
            <button 
              onClick={buildAndTrainModel}
              disabled={!dataset || modelStatus === 'training'}
              className="w-full mb-4 flex items-center justify-center gap-2 bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-500 hover:to-blue-500 disabled:opacity-50 disabled:cursor-not-allowed text-white py-2 px-4 rounded-lg transition-colors shadow-lg"
            >
              {modelStatus === 'training' ? <RefreshCw size={16} className="animate-spin" /> : <Play size={16} />}
              2. Train CNN
            </button>

            {modelStatus === 'training' && (
              <div className="mt-4">
                <div className="flex justify-between text-xs text-slate-400 mb-1">
                  <span>Progress</span>
                  <span>{Math.round(progress)}%</span>
                </div>
                <div className="w-full bg-slate-700 rounded-full h-1.5 overflow-hidden">
                  <div className="bg-yellow-400 h-1.5 transition-all duration-300" style={{ width: `${progress}%` }}></div>
                </div>
                <p className="text-xs text-slate-500 mt-2 font-mono text-center">{epochInfo}</p>
              </div>
            )}
          </div>

          {/* Drawing Input Canvas */}
          <div className="flex flex-col items-center gap-4 bg-slate-800 p-6 rounded-xl border border-slate-700 shadow-xl w-full max-w-sm">
            <div className="flex items-center gap-2 mb-2 w-full justify-between">
              <h3 className="text-slate-200 font-semibold flex items-center gap-2">
                <PenTool size={18} className="text-cyan-400" />
                Input Stimulus
              </h3>
              <button 
                onClick={clearCanvas}
                className="text-slate-400 hover:text-white transition-colors"
                title="Clear Pad"
              >
                <Eraser size={18} />
              </button>
            </div>
            
            <canvas
              ref={canvasRef}
              width={200}
              height={200}
              onMouseDown={startDrawing}
              onMouseMove={draw}
              onMouseUp={stopDrawing}
              onMouseOut={stopDrawing}
              onTouchStart={startDrawing}
              onTouchMove={draw}
              onTouchEnd={stopDrawing}
              style={{ touchAction: 'none' }}
              className={`bg-black border-2 border-slate-600 rounded-lg cursor-crosshair touch-none ${modelStatus !== 'trained' ? 'opacity-50 pointer-events-none' : 'shadow-[0_0_15px_rgba(34,211,238,0.2)]'}`}
            />
            
            <div className="w-full mt-4">
              <h4 className="text-slate-300 text-sm mb-3 uppercase tracking-wider font-bold">Network Output</h4>
              <div className="grid grid-cols-2 gap-x-4 gap-y-2">
                {CLASSES.map((cls, idx) => (
                  <div key={cls} className="mb-1">
                    <div className="flex justify-between text-[10px] text-slate-400 mb-1">
                      <span>Digit {cls}</span>
                      <span className="font-mono">{(predictions[idx] * 100).toFixed(1)}%</span>
                    </div>
                    <div className="w-full bg-slate-700 rounded-full h-1.5 overflow-hidden">
                      <div 
                        className="bg-cyan-400 h-1.5 transition-all duration-300" 
                        style={{ width: `${predictions[idx] * 100}%` }}
                      ></div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

        </div>

        {/* Right Column: Visualizer */}
        <div className="lg:col-span-3 bg-slate-800 p-6 rounded-xl border border-slate-700 shadow-xl flex flex-col">
          <div className="flex items-center justify-between border-b border-slate-700 pb-4 mb-6">
            <h3 className="text-xl font-semibold flex items-center gap-2">
              <Eye size={20} className="text-cyan-400" />
              Network Graph Visualization
            </h3>
            <span className="text-xs text-slate-400 bg-slate-900 px-3 py-1 rounded-full border border-slate-700">
              Hover nodes to trace connections
            </span>
          </div>

          {!Object.keys(layerVis).length ? (
            <div className="flex-grow flex flex-col items-center justify-center text-slate-500 min-h-[400px]">
              <BrainCircuit size={64} className="opacity-20 mb-4" />
              <p>Train the model and draw a digit (0-9) to visualize internal layers.</p>
            </div>
          ) : (
            (() => {
              const NODE_SIZE = 50;
              const NODE_GAP = 16;
              const LAYER_SPACING = 180;
              const CANVAS_WIDTH = 1200;

              const layers = Object.entries(layerVis);
              const maxNodes = Math.max(...layers.map(([_, maps]) => maps.length));
              const CANVAS_HEIGHT = Math.max(600, maxNodes * (NODE_SIZE + NODE_GAP) + 100);
              
              // Calculate Node Layout Positions
              const nodePositions = layers.map(([name, maps], layerIdx) => {
                const numNodes = maps.length;
                const startY = (CANVAS_HEIGHT - (numNodes * NODE_SIZE + (numNodes - 1) * NODE_GAP)) / 2;
                const x = 60 + layerIdx * LAYER_SPACING;
                
                return maps.map((map, nodeIdx) => ({
                  id: `${layerIdx}-${nodeIdx}`,
                  layerIdx,
                  nodeIdx,
                  x,
                  y: startY + nodeIdx * (NODE_SIZE + NODE_GAP),
                  cx: x + NODE_SIZE / 2,
                  cy: startY + nodeIdx * (NODE_SIZE + NODE_GAP) + NODE_SIZE / 2,
                  map,
                  layerName: name
                }));
              });

              // Calculate Connections (Conv layers are fully connected, MaxPool are 1-to-1)
              const edges = [];
              for (let l = 0; l < layers.length - 1; l++) {
                const currentNodes = nodePositions[l];
                const nextNodes = nodePositions[l + 1];
                const isOneToOne = (l === 1 || l === 3); 
                
                currentNodes.forEach(sourceNode => {
                  nextNodes.forEach(targetNode => {
                    if (isOneToOne && sourceNode.nodeIdx !== targetNode.nodeIdx) return;
                    edges.push({
                      id: `${sourceNode.id}-to-${targetNode.id}`,
                      source: sourceNode,
                      target: targetNode
                    });
                  });
                });
              }

              return (
                <div className="relative w-full overflow-x-auto bg-slate-900 rounded-xl border border-slate-700 shadow-inner">
                  <div style={{ width: CANVAS_WIDTH, height: CANVAS_HEIGHT }} className="relative mx-auto min-w-[900px]">
                    {/* Layer Labels */}
                    {layers.map(([name], idx) => (
                      <div 
                        key={name}
                        className="absolute text-center w-40 text-xs font-semibold text-slate-400 top-6 -translate-x-1/2 bg-slate-900/80 px-2 py-1 rounded-full border border-slate-700 z-0"
                        style={{ left: 60 + idx * LAYER_SPACING + NODE_SIZE / 2 }}
                      >
                        {name}
                      </div>
                    ))}

                    {/* Connection Edges (SVG) */}
                    <svg className="absolute inset-0 w-full h-full pointer-events-none z-0">
                      {edges.map(edge => {
                        const isHovered = hoveredNode === edge.source.id || hoveredNode === edge.target.id;
                        const isDimmed = hoveredNode && !isHovered;
                        const x1 = edge.source.x + NODE_SIZE;
                        const y1 = edge.source.cy;
                        const x2 = edge.target.x;
                        const y2 = edge.target.cy;
                        
                        // Cubic bezier control points for beautiful smooth curves
                        const cpX1 = x1 + 50;
                        const cpX2 = x2 - 50;

                        return (
                          <path 
                            key={edge.id}
                            d={`M ${x1} ${y1} C ${cpX1} ${y1}, ${cpX2} ${y2}, ${x2} ${y2}`}
                            fill="none"
                            stroke={isHovered ? '#22d3ee' : '#334155'}
                            strokeWidth={isHovered ? 2.5 : 1}
                            className={`transition-all duration-300 ${isDimmed ? 'opacity-10' : 'opacity-70'}`}
                          />
                        )
                      })}
                    </svg>
                    
                    {/* Node Images (Feature Maps) */}
                    {nodePositions.map((layer) => 
                      layer.map(node => {
                        const isHovered = hoveredNode === node.id;
                        const isDimmed = hoveredNode && !isHovered;
                        
                        return (
                          <div 
                            key={node.id}
                            className={`absolute transition-all duration-300 z-10 flex flex-col items-center justify-center cursor-crosshair
                              ${isHovered ? 'scale-150 z-30' : 'scale-100 z-20'} 
                              ${isDimmed ? 'opacity-30' : 'opacity-100'}
                            `}
                            style={{ left: node.x, top: node.y, width: NODE_SIZE, height: NODE_SIZE }}
                            onMouseEnter={() => setHoveredNode(node.id)}
                            onMouseLeave={() => setHoveredNode(null)}
                          >
                            <img 
                              src={node.map.url}
                              alt={`filter-${node.nodeIdx}`}
                              className={`w-full h-full border rounded bg-black pixelated shadow-[0_4px_15px_rgba(0,0,0,0.5)] transition-colors duration-300
                                ${node.map.isDense ? 'object-fill' : 'object-contain'}
                                ${isHovered ? 'border-cyan-400 shadow-cyan-500/20' : 'border-slate-600'}
                              `}
                              style={{ imageRendering: 'pixelated' }}
                            />
                            {/* Hover info tooltip */}
                            <div 
                              className={`absolute top-[110%] bg-slate-950 border border-cyan-500/50 text-[8px] text-cyan-300 px-2 py-1 rounded whitespace-nowrap pointer-events-none transition-opacity duration-200 shadow-lg ${isHovered ? 'opacity-100' : 'opacity-0'}`}
                            >
                              F{node.nodeIdx} [{node.map.min}, {node.map.max}]
                            </div>
                          </div>
                        );
                      })
                    )}
                  </div>
                </div>
              );
            })()
          )}
        </div>

      </div>
    </div>
  );
}
