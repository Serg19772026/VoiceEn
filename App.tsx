
import React, { useState, useEffect, useRef, useCallback } from 'react';
import { GoogleGenAI, Modality, LiveServerMessage } from '@google/genai';
import { GoogleGenerativeAI } from "@google/generative-ai";
import { 
  Mic, Languages, MessageSquare, Volume2, 
  Keyboard, Send, Wifi, WifiOff, X, Settings, Trash2, AlertCircle, Lock, Unlock, Square, ArrowLeftRight
} from 'lucide-react';
import { TranslationMode, ChatMessage } from './types';
import { decode, encode, decodeAudioData, createPcmBlob } from './utils/audioUtils';

const App: React.FC = () => {
  const [mode, setMode] = useState<TranslationMode>(TranslationMode.EN_TO_RU);
  const [isRecording, setIsRecording] = useState(false);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [status, setStatus] = useState<'Idle' | 'Connecting' | 'Live'>('Idle');
  const [isOnline, setIsOnline] = useState(navigator.onLine);
  const [showKeyboard, setShowKeyboard] = useState(false);
  const [inputText, setInputText] = useState('');
  const [isTranslatingText, setIsTranslatingText] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [isMasterEnabled, setIsMasterEnabled] = useState(true);
  const [inputVolume, setInputVolume] = useState(0);
  
  const [liveTranscription, setLiveTranscription] = useState<{user: string, model: string}>({user: '', model: ''});
  
  const sessionRef = useRef<any>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const outputAudioContextRef = useRef<AudioContext | null>(null);
  const nextStartTimeRef = useRef<number>(0);
  const sourcesRef = useRef<Set<AudioBufferSourceNode>>(new Set());
  const streamRef = useRef<MediaStream | null>(null);
  const transcriptionBufferRef = useRef({ user: '', model: '' });
  const analyserRef = useRef<AnalyserNode | null>(null);
  const animationFrameRef = useRef<number>(0);

  useEffect(() => {
    const handleOnline = () => setIsOnline(true);
    const handleOffline = () => setIsOnline(false);
    window.addEventListener('online', handleOnline);
    window.addEventListener('offline', handleOffline);
    return () => {
      window.removeEventListener('online', handleOnline);
      window.removeEventListener('offline', handleOffline);
      if (animationFrameRef.current) cancelAnimationFrame(animationFrameRef.current);
    };
  }, []);

  const getSystemInstruction = (m: TranslationMode) => {
    const target = m === TranslationMode.EN_TO_RU ? 'Russian' : 'English';
    const source = m === TranslationMode.EN_TO_RU ? 'English' : 'Russian';
    
    return `You are a professional real-time voice translator between English and Russian.
STRICT RULES:
1. ONLY output the direct ${target} translation.
2. NEVER repeat, echo, or include the user's original ${source} words in your response or transcription.
3. DO NOT include any conversational filler, explanations, or labels.
4. If you hear noise or unintelligible audio, output NOTHING.
5. Provide text transcription ONLY for the translated ${target} text.`;
  };

  const addMessage = (sender: 'user' | 'model', text: string, isFromKeyboard: boolean = false) => {
    let trimmed = text.trim();
    if (!trimmed) return;
    
    // Filter noise artifacts
    const noisePatterns = [/^<.*>$/i, /^\d+$/, /^[.,!?;: ]+$/, /^noise$/i, /^static$/i];
    if (!isFromKeyboard && noisePatterns.some(pattern => pattern.test(trimmed))) return;

    setMessages(prev => {
      if (prev.length > 0) {
        const lastMsg = prev[prev.length - 1];
        if (lastMsg.text.toLowerCase() === trimmed.toLowerCase() && lastMsg.sender === sender) {
          return prev;
        }
      }
      return [
        ...prev,
        { id: Date.now().toString() + Math.random(), sender, text: trimmed, timestamp: Date.now() }
      ];
    });
  };

  const stopSession = useCallback(() => {
    if (sessionRef.current) {
      sessionRef.current.close();
      sessionRef.current = null;
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
    sourcesRef.current.forEach(s => { try { s.stop(); } catch (e) {} });
    sourcesRef.current.clear();
    nextStartTimeRef.current = 0;
    setIsRecording(false);
    setStatus('Idle');
    setInputVolume(0);
    setLiveTranscription({user: '', model: ''});
    transcriptionBufferRef.current = { user: '', model: '' };
    if (animationFrameRef.current) cancelAnimationFrame(animationFrameRef.current);
  }, []);

  const updateVolume = () => {
    if (!analyserRef.current) return;
    const dataArray = new Uint8Array(analyserRef.current.frequencyBinCount);
    analyserRef.current.getByteFrequencyData(dataArray);
    const sum = dataArray.reduce((a, b) => a + b, 0);
    const avg = sum / dataArray.length;
    setInputVolume(avg);
    animationFrameRef.current = requestAnimationFrame(updateVolume);
  };

  const startSession = async (selectedMode: TranslationMode) => {
    if (!isOnline || !isMasterEnabled) return;
    if (isRecording) {
      const wasSameMode = mode === selectedMode;
      stopSession();
      if (wasSameMode) return;
    }
    setMode(selectedMode);
    
    try {
      setStatus('Connecting');
      if (!audioContextRef.current) audioContextRef.current = new AudioContext({ sampleRate: 16000 });
      if (!outputAudioContextRef.current) outputAudioContextRef.current = new AudioContext({ sampleRate: 24000 });
      await audioContextRef.current.resume();
      await outputAudioContextRef.current.resume();

      const ai = new GoogleGenAI({ apiKey: import.meta.env.VITE_API_KEY});
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;
      const sourceNode = audioContextRef.current.createMediaStreamSource(stream);
      const analyser = audioContextRef.current.createAnalyser();
      analyser.fftSize = 256;
      sourceNode.connect(analyser);
      analyserRef.current = analyser;
      updateVolume();

      const sessionPromise = ai.live.connect({
        model: 'gemini-2.5-flash-native-audio-preview-09-2025',
        config: {
          responseModalities: [Modality.AUDIO],
          speechConfig: {
            voiceConfig: { prebuiltVoiceConfig: { voiceName: selectedMode === TranslationMode.EN_TO_RU ? 'Kore' : 'Zephyr' } },
          },
          systemInstruction: getSystemInstruction(selectedMode),
          inputAudioTranscription: {},
          outputAudioTranscription: {},
        },
        callbacks: {
          onopen: () => {
            setStatus('Live');
            setIsRecording(true);
            const scriptProcessor = audioContextRef.current!.createScriptProcessor(4096, 1, 1);
            
            scriptProcessor.onaudioprocess = (e) => {
              // ПРЕДОХРАНИТЕЛЬ №1: Если мы вручную выключили запись или сессии больше нет
              if (!sessionRef.current) {
                sourceNode.disconnect(scriptProcessor);
                scriptProcessor.disconnect();
                return;
              }

              const pcmBlob = createPcmBlob(e.inputBuffer.getChannelData(0));
              
              sessionPromise.then(s => {
                // ПРЕДОХРАНИТЕЛЬ №2: Проверяем, не закрылась ли сессия за ту миллисекунду, пока мы готовили звук
                if (s && sessionRef.current) {
                  try {
                    s.sendRealtimeInput({ media: pcmBlob });
                  } catch (err) {
                    // Если поймали ошибку отправки - тихо выключаемся без спама
                    stopSession();
                  }
                }
              });
            };
            
            sourceNode.connect(scriptProcessor);
            scriptProcessor.connect(audioContextRef.current!.destination);
          },
          onmessage: async (m: LiveServerMessage) => {
            const audioData = m.serverContent?.modelTurn?.parts[0]?.inlineData?.data;
            if (audioData) {
              const outCtx = outputAudioContextRef.current!;
              nextStartTimeRef.current = Math.max(nextStartTimeRef.current, outCtx.currentTime);
              const audioBuffer = await decodeAudioData(decode(audioData), outCtx, 24000, 1);
              const source = outCtx.createBufferSource();
              source.buffer = audioBuffer;
              source.connect(outCtx.destination);
              source.addEventListener('ended', () => sourcesRef.current.delete(source));
              source.start(nextStartTimeRef.current);
              nextStartTimeRef.current += audioBuffer.duration;
              sourcesRef.current.add(source);
            }
            if (m.serverContent?.inputTranscription) {
              transcriptionBufferRef.current.user += m.serverContent.inputTranscription.text;
              setLiveTranscription(prev => ({ ...prev, user: transcriptionBufferRef.current.user }));
            }
            if (m.serverContent?.outputTranscription) {
              transcriptionBufferRef.current.model += m.serverContent.outputTranscription.text;
              setLiveTranscription(prev => ({ ...prev, model: transcriptionBufferRef.current.model }));
            }
            if (m.serverContent?.turnComplete) {
              const uText = transcriptionBufferRef.current.user.trim();
              let mText = transcriptionBufferRef.current.model.trim();
              
              if (uText && mText.toLowerCase().startsWith(uText.toLowerCase())) {
                mText = mText.substring(uText.length).trim();
                mText = mText.replace(/^[.,!?;: ]+/, '');
              }

              if (uText) addMessage('user', uText);
              if (mText) addMessage('model', mText);
              
              transcriptionBufferRef.current = { user: '', model: '' };
              setLiveTranscription({ user: '', model: '' });
            }
          },
          onerror: (e) => { 
            console.error("ОШИБКА ГОЛОСОВОГО КАНАЛА:", e);
            stopSession(); 
          },
          onclose: (reason) => {
            console.log("ГОЛОСОВОЙ КАНАЛ ЗАКРЫТ. Причина:", reason);
            stopSession();
          }
        }
      });
      sessionRef.current = await sessionPromise;
    } catch (e) { setStatus('Idle'); }
  };

  const handleMasterToggle = () => {
    setIsMasterEnabled(p => {
      if (p && isRecording) stopSession();
      return !p;
    });
  };

    const handleTextTranslate = async () => {
    if (!inputText.trim()) return;
    const text = inputText;
    setInputText('');
    setShowKeyboard(false);
    addMessage('user', text, true);
    setIsTranslatingText(true);

    const targetLangName = mode === TranslationMode.EN_TO_RU ? 'Russian' : 'English';

    try {
      const aiObj = (window as any).ai;
      const localModel = aiObj?.languageModel || (window as any).LanguageModel;

      if (localModel) {
        console.log("Локальный ИИ найден. Начинаю быстрый перевод...");
        const session = await localModel.create(); // Создаем пустую сессию без сложного системного промпта
        
        // Передаем инструкцию прямо внутри промпта - это работает в 100% случаев
        const translation = await session.prompt(
          `Translate the following text into ${targetLangName}. Output ONLY the translation: ${text}`
        );
        
        session.destroy();
        console.log("Перевод готов:", translation);
        
        addMessage('model', translation.trim(), true);
        speakText(translation, mode === TranslationMode.EN_TO_RU ? 'ru-RU' : 'en-US');
        setIsTranslatingText(false);
        return; 
      }

      // --- ОБЛАЧНЫЙ FALLBACK (если локальный ИИ всё же не сработал) ---
      const genAI = new GoogleGenerativeAI(import.meta.env.VITE_API_KEY);
      const model = genAI.getGenerativeModel({ model: "gemini-2.5-flash" });
      const result = await model.generateContent(`Translate to ${targetLangName}: ${text}. Output ONLY translation.`);
      const cloudTranslation = result.response.text().trim();

      addMessage('model', cloudTranslation, true);
      speakText(cloudTranslation, mode === TranslationMode.EN_TO_RU ? 'ru-RU' : 'en-US');

    } catch (err) {
      console.error("Ошибка:", err);
      addMessage('model', "Connection error. Check VPN.");
    } finally {
      setIsTranslatingText(false);
    }
  };

  const speakText = (text: string, lang?: string) => {
    if (!text || text.length < 1) return;
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = lang || (/[а-яА-Я]/.test(text) ? 'ru-RU' : 'en-US');
    window.speechSynthesis.speak(utterance);
  };

  const chatEndRef = useRef<HTMLDivElement>(null);
  useEffect(() => { chatEndRef.current?.scrollIntoView({ behavior: 'smooth' }); }, [messages, liveTranscription]);

  const VolumeBars = () => {
    const bars = [1, 2, 3, 2];
    const scaledVol = Math.max(0, (inputVolume - 10) * 1.5);
    return (
      <div className="absolute -top-12 left-1/2 -translate-x-1/2 flex items-end gap-1.5 h-10 bg-indigo-600/90 backdrop-blur-xl px-4 py-3 rounded-full border border-white/30 shadow-xl">
        {bars.map((m, i) => (
          <div key={i} className="w-1.5 bg-white rounded-full transition-all duration-75" style={{ height: `${Math.min(100, Math.max(15, (scaledVol / 60) * 100 * m))}%` }} />
        ))}
      </div>
    );
  };

  return (
    <div className="flex flex-col h-screen bg-slate-100 text-slate-900 font-sans overflow-hidden">
      <header className="bg-white px-5 py-3.5 shadow-sm flex items-center justify-between z-10 border-b border-slate-200">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 bg-indigo-600 rounded-lg flex items-center justify-center text-white shadow-md">
            <Languages className="w-5 h-5" />
          </div>
          <h1 className="text-lg font-black tracking-tight text-slate-800">Voice Match</h1>
        </div>
        <div className="flex items-center gap-2">
          <div className={`flex items-center gap-1.5 px-3 py-1 rounded-full text-[10px] font-black uppercase tracking-widest ${isOnline ? 'bg-green-100 text-green-700' : 'bg-red-100 text-red-700'}`}>
            {isOnline ? <Wifi className="w-3 h-3" /> : <WifiOff className="w-3 h-3" />}
            {isOnline ? 'Online' : 'Offline'}
          </div>
          <button onClick={() => setShowSettings(true)} className="p-2 hover:bg-slate-100 rounded-full text-slate-400 transition-colors">
            <Settings className="w-5 h-5" />
          </button>
        </div>
      </header>

      <main className="flex-1 overflow-y-auto px-4 space-y-3 pb-64 pt-6 scroll-smooth bg-slate-50/50">
        {messages.length === 0 && !liveTranscription.user && !liveTranscription.model ? (
          <div className="h-full flex flex-col items-center justify-center text-slate-300 space-y-4">
            <div className="w-20 h-20 bg-white rounded-full flex items-center justify-center shadow-inner border border-slate-100">
              <MessageSquare className="w-8 h-8 opacity-10" />
            </div>
            <p className="text-[10px] font-black text-center px-16 leading-relaxed opacity-40 uppercase tracking-[0.2em]">
              Select a flag and start speaking
            </p>
          </div>
        ) : (
          <>
            {messages.map((msg) => (
              <div key={msg.id} className="flex flex-col items-center animate-in slide-in-from-bottom-2 duration-300">
                <div className={`max-w-[95%] px-5 py-3 rounded-[1.5rem] shadow-sm border text-center transition-all ${
                  msg.sender === 'user' ? 'bg-white border-slate-200 text-slate-800' : 'bg-indigo-600 border-indigo-500 text-white'
                }`}>
                  <div className="flex items-center justify-center gap-3">
                     <p className="text-[15px] leading-snug font-bold tracking-tight">{msg.text}</p>
                     <button onClick={() => speakText(msg.text)} className={`p-1 rounded-full ${msg.sender === 'user' ? 'text-slate-300' : 'text-white/40'}`}>
                      <Volume2 className="w-4 h-4" />
                     </button>
                  </div>
                </div>
              </div>
            ))}
            {liveTranscription.user && (
              <div className="flex flex-col items-center opacity-40">
                <div className="max-w-[95%] px-5 py-2.5 rounded-2xl border border-dashed border-slate-300 bg-white/40 text-slate-500 text-center italic">
                  <p className="text-[14px] leading-tight font-medium">{liveTranscription.user}</p>
                </div>
              </div>
            )}
            {liveTranscription.model && (
              <div className="flex flex-col items-center opacity-40">
                <div className="max-w-[95%] px-5 py-2.5 rounded-2xl border border-dashed border-indigo-300 bg-indigo-50/50 text-indigo-500 text-center italic">
                  <p className="text-[14px] leading-tight font-medium">{liveTranscription.model}</p>
                </div>
              </div>
            )}
          </>
        )}
        <div ref={chatEndRef} />
      </main>

      <div className="fixed bottom-0 inset-x-0 p-6 z-20 pointer-events-none">
        <div className="max-w-md mx-auto flex flex-col items-center gap-4 pointer-events-auto">
          
          {showKeyboard && (
            <div className="w-full bg-white border border-slate-200 rounded-[2rem] p-3 shadow-2xl flex flex-col animate-in slide-in-from-bottom-4 ring-1 ring-black/5 overflow-hidden">
              <div className="flex items-center justify-between px-6 py-4 border-b border-slate-50 bg-slate-50/50">
                <div className="flex items-center gap-6 w-full">
                  <span className={`text-[11px] font-black uppercase tracking-wider transition-all duration-300 flex-1 text-center ${mode === TranslationMode.EN_TO_RU ? 'text-indigo-600' : 'text-slate-400 opacity-60'}`}>
                    English
                  </span>
                  
                  <button 
                    onClick={() => setMode(m => m === TranslationMode.EN_TO_RU ? TranslationMode.RU_TO_EN : TranslationMode.EN_TO_RU)}
                    className="group relative w-12 h-12 flex items-center justify-center bg-indigo-600 hover:bg-indigo-700 text-white rounded-full shadow-lg shadow-indigo-100 hover:scale-110 active:scale-90 transition-all"
                    title="Change direction"
                  >
                    <ArrowLeftRight className={`w-5 h-5 transition-transform duration-500 ${mode === TranslationMode.RU_TO_EN ? 'rotate-180' : ''}`} />
                    <div className="absolute inset-0 rounded-full bg-white/20 scale-0 group-active:scale-100 transition-transform duration-200"></div>
                  </button>

                  <span className={`text-[11px] font-black uppercase tracking-wider transition-all duration-300 flex-1 text-center ${mode === TranslationMode.RU_TO_EN ? 'text-indigo-600' : 'text-slate-400 opacity-60'}`}>
                    Russian
                  </span>
                </div>
              </div>
              <div className="flex items-end gap-2 p-1">
                <textarea 
                  autoFocus
                  value={inputText}
                  onChange={(e) => setInputText(e.target.value)}
                  placeholder={mode === TranslationMode.EN_TO_RU ? "Enter text in English..." : "Введите текст на русском..."}
                  className="flex-1 bg-transparent border-none focus:ring-0 p-4 text-base min-h-[56px] max-h-[150px] resize-none font-bold"
                  onKeyDown={(e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleTextTranslate(); } }}
                />
                <button onClick={handleTextTranslate} className="w-12 h-12 flex items-center justify-center bg-indigo-600 text-white rounded-full shadow-lg mb-1 mr-1">
                  <Send className="w-5 h-5" />
                </button>
              </div>
            </div>
          )}

          <div className="bg-white/95 backdrop-blur-2xl border border-white shadow-[0_20px_50px_rgba(0,0,0,0.15)] rounded-[3.5rem] p-3 flex items-center justify-between w-full ring-1 ring-slate-200/50">
            <button
              onClick={() => startSession(TranslationMode.EN_TO_RU)}
              disabled={!isMasterEnabled}
              className={`relative w-16 h-16 rounded-full flex items-center justify-center transition-all duration-300 ${!isOnline || !isMasterEnabled ? 'grayscale opacity-10' : 'hover:scale-105 active:scale-95'} ${isRecording && mode === TranslationMode.EN_TO_RU ? 'ring-[6px] ring-indigo-500/30 scale-110 shadow-2xl' : 'shadow-lg'}`}
            >
              {isRecording && mode === TranslationMode.EN_TO_RU && <VolumeBars />}
              <div className="absolute inset-0 rounded-full overflow-hidden border border-black/10">
                <div className="absolute inset-0 flex flex-col">
                  <div className="h-1/2 bg-[#002664] w-1/2 z-10" />
                  <div className="absolute inset-0 flex flex-col">
                    {[...Array(13)].map((_, i) => <div key={i} className={`h-[7.7%] ${i % 2 === 0 ? 'bg-[#BD3D44]' : 'bg-white'}`} />)}
                  </div>
                </div>
              </div>
              <div className="relative z-10">{isRecording && mode === TranslationMode.EN_TO_RU ? <Square className="w-6 h-6 text-white fill-white" /> : <Mic className="w-7 h-7 text-white" />}</div>
              <div className="absolute bottom-1 right-2 text-[7px] font-black text-white/60 drop-shadow-md">EN</div>
            </button>

            <div className="flex items-center gap-3">
              <button onClick={handleMasterToggle} className={`w-12 h-12 rounded-full flex items-center justify-center border-2 transition-all ${isMasterEnabled ? 'bg-green-50 text-green-600 border-green-200 shadow-sm' : 'bg-red-50 text-red-600 border-red-200 shadow-inner'}`}>
                {isMasterEnabled ? <Unlock className="w-5 h-5" /> : <Lock className="w-5 h-5" />}
              </button>
              <button onClick={() => { setShowKeyboard(!showKeyboard); if (isRecording) stopSession(); }} className={`w-12 h-12 rounded-full flex items-center justify-center ${showKeyboard ? 'bg-indigo-600 text-white shadow-indigo-200' : 'bg-slate-50 text-slate-400 border border-slate-100'}`}>
                {showKeyboard ? <X className="w-5 h-5" /> : <Keyboard className="w-5 h-5" />}
              </button>
            </div>

            <button
              onClick={() => startSession(TranslationMode.RU_TO_EN)}
              disabled={!isMasterEnabled}
              className={`relative w-16 h-16 rounded-full flex items-center justify-center transition-all duration-300 ${!isOnline || !isMasterEnabled ? 'grayscale opacity-10' : 'hover:scale-105 active:scale-95'} ${isRecording && mode === TranslationMode.RU_TO_EN ? 'ring-[6px] ring-indigo-500/30 scale-110 shadow-2xl' : 'shadow-lg'}`}
            >
              {isRecording && mode === TranslationMode.RU_TO_EN && <VolumeBars />}
              <div className="absolute inset-0 rounded-full overflow-hidden border border-black/10">
                <div className="absolute inset-0 flex flex-col"><div className="h-1/3 bg-white" /><div className="h-1/3 bg-[#0039A6]" /><div className="h-1/3 bg-[#D52B1E]" /></div>
              </div>
              <div className="relative z-10">{isRecording && mode === TranslationMode.RU_TO_EN ? <Square className="w-6 h-6 text-white fill-white" /> : <Mic className="w-7 h-7 text-white" />}</div>
              <div className="absolute bottom-1 right-2 text-[7px] font-black text-white/60 drop-shadow-md">RU</div>
            </button>
          </div>
        </div>
      </div>

      {showSettings && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-900/40 backdrop-blur-sm px-4">
          <div className="w-full max-sm bg-white rounded-[2rem] shadow-2xl overflow-hidden animate-in zoom-in-95">
            <div className="p-6 border-b border-slate-50 flex items-center justify-between">
              <h2 className="text-xl font-black text-slate-800">Translator Settings</h2>
              <button onClick={() => setShowSettings(false)} className="p-2 bg-slate-50 rounded-full"><X className="w-5 h-5" /></button>
            </div>
            <div className="p-6 space-y-4">
              <button onClick={() => { setMessages([]); setShowSettings(false); }} className="w-full py-4 bg-slate-50 text-slate-800 rounded-2xl font-black text-sm border border-slate-100 hover:bg-red-50 hover:text-red-600 transition-colors flex items-center justify-center gap-2">
                <Trash2 className="w-4 h-4" /> Clear Translation History
              </button>
              <div className="p-4 bg-slate-50 rounded-2xl border border-slate-100">
                <p className="text-[10px] font-black text-slate-400 uppercase mb-2 tracking-widest">Engine Status</p>
                <div className="flex justify-between text-[12px] font-bold">
                  <span>Gemini 2.5 Live</span>
                  <span className={isRecording ? 'text-green-600' : 'text-slate-400'}>{isRecording ? 'STREAMS ACTIVE' : 'IDLE'}</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default App;
