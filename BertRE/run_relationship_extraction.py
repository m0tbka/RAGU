import pandas as pd
from itertools import combinations
from typing import List, Dict

# Предполагается, что ваш класс модели RelationClassificationModel определен
# и доступен для импорта.
from relation_classifier import RelationClassificationModel


def extract_relationships(
    df_chunks: pd.DataFrame, 
    df_entities: pd.DataFrame, 
    model: RelationClassificationModel, 
    batch_size: int = 32, 
    filter_none: bool = True
) -> pd.DataFrame:
    """
    Извлекает отношения между сущностями в текстовых чанках с помощью обученной модели,
    эффективно обрабатывая данные в пакетах (батчах).

    Args:
        df_chunks (pd.DataFrame): DataFrame с колонками ['chunk', 'chunk_id'].
        df_entities (pd.DataFrame): DataFrame с колонками ['entity_name', 'entity_type', 'start', 'end', 'chunk_id'].
        model (RelationClassificationModel): Инициализированный экземпляр вашей обученной модели.
        batch_size (int): Размер пакета для отправки в модель. Подберите под вашу видеокарту (e.g., 16, 32, 64).
        filter_none (bool): Если True, отношения типа "NONE" не будут включены в результат.

    Returns:
        pd.DataFrame: Итоговый DataFrame с извлеченными отношениями и колонками
                      ["source_entity", "target_entity", "relationship_type", 
                       "relationship_description", "relationship_strength", "chunk_id"].
    """
    
    # 1. Подготовка данных: создаем словарь для быстрого доступа к тексту чанка
    chunk_text_map = pd.Series(df_chunks.chunk.values, index=df_chunks.chunk_id).to_dict()
    
    # 2. Генерация всех пар сущностей для последующей обработки
    all_pairs_to_process: List[Dict] = []
    
    grouped_entities = df_entities.groupby('chunk_id')
    
    for chunk_id, group in grouped_entities:
        if len(group) < 2:
            continue
            
        chunk_text = chunk_text_map.get(chunk_id)
        if not chunk_text:
            continue

        entities_in_chunk = group.sort_values('start').to_dict('records')
        
        # Создаем все уникальные комбинации из 2-х сущностей
        for entity1, entity2 in combinations(entities_in_chunk, 2):
            e1, e2 = entity1, entity2  # Уже отсортированы по 'start'
            
            try:
                # Собираем размеченный текст из частей, чтобы избежать ошибок с индексами
                parts = [
                    chunk_text[:e1['start']],
                    "<e1>", chunk_text[e1['start']:e1['end']], "</e1>",
                    chunk_text[e1['end']:e2['start']],
                    "<e2>", chunk_text[e2['start']:e2['end']], "</e2>",
                    chunk_text[e2['end']:]
                ]
                formatted_text = "".join(parts)
                
                # Сохраняем всю необходимую информацию для последующей обработки
                all_pairs_to_process.append({
                    "formatted_text": formatted_text,
                    "source_entity": e1['entity_name'],
                    "target_entity": e2['entity_name'],
                    "chunk_id": chunk_id
                })
            except (IndexError, TypeError):
                # Пропускаем пару, если индексы start/end некорректны
                print(f"Внимание: Пропущена пара в chunk_id={chunk_id} из-за некорректных индексов сущностей.")
                continue

    if not all_pairs_to_process:
        # Возвращаем пустой DataFrame с нужными колонками, если пар не найдено
        return pd.DataFrame(columns=["source_entity", "target_entity", "relationship_type", 
                                     "relationship_description", "relationship_strength", "chunk_id"])

    # 3. Пакетное предсказание отношений
    all_relationships: List[Dict] = []
    
    for i in range(0, len(all_pairs_to_process), batch_size):
        batch_data = all_pairs_to_process[i:i + batch_size]
        texts_to_predict = [item['formatted_text'] for item in batch_data]
        
        predictions = model.predict(texts_to_predict)
        
        for j, prediction in enumerate(predictions):
            if filter_none and prediction == "NONE":
                continue
            
            original_pair_data = batch_data[j]
            all_relationships.append({
                "source_entity": original_pair_data['source_entity'],
                "target_entity": original_pair_data['target_entity'],
                "relationship_type": prediction,
                "relationship_description": None,
                "relationship_strength": None,
                "chunk_id": original_pair_data['chunk_id']
            })
            
    # 4. Возвращаем итоговый DataFrame
    return pd.DataFrame(all_relationships)
