
export enum TranslationMode {
  EN_TO_RU = 'EN_TO_RU',
  RU_TO_EN = 'RU_TO_EN'
}

export interface ChatMessage {
  id: string;
  sender: 'user' | 'model';
  text: string;
  timestamp: number;
}
