import json
import re
import logging
from websocietysimulator import Simulator
from websocietysimulator.agent import RecommendationAgent
from websocietysimulator.llm import LLMBase
import tiktoken

logging.basicConfig(level=logging.INFO)

def num_tokens_from_string(string: str) -> int:
    """计算文本的token数量"""
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(string))

def get_structured_item_info(item):
    """结构化处理不同来源的item信息"""
    source = item.get('source', '')
    info = {'item_id': item['item_id']}
    
    # Yelp商家信息
    if source == 'yelp':
        info['type'] = 'business'
        info['name'] = item.get('name', '')
        info['rating'] = item.get('stars', 0)
        info['categories'] = item.get('categories', '')
        attrs = []
        for k, v in item.get('attributes', {}).items():
            cleaned_v = str(v).replace("'", "").replace("{", "").replace("}", "")
            attrs.append(f"{k.replace('_', ' ').title()}: {cleaned_v}")
        info['features'] = ' | '.join(attrs)
    
    # Amazon商品
    elif source == 'amazon':
        info['type'] = 'product'
        info['title'] = item.get('title', '')
        info['brand'] = item.get('details', {}).get('Brand', 'Unknown')
        info['features'] = ' • '.join(item.get('features', []))
        info['price'] = item.get('price', 'N/A')
    
    # Goodreads书籍
    elif source == 'goodreads':
        info['type'] = 'book'
        info['title'] = item.get('title_without_series', item.get('title', ''))
        info['author'] = item.get('authors', [{}])[0].get('author_id', 'Unknown')
        info['rating'] = float(item.get('average_rating', 0))
    
    # 通用字段处理
    info['source'] = source
    info['review_count'] = item.get('review_count') or item.get('rating_number') or item.get('ratings_count', 0)
    return info

def generate_user_profile(user_id, review_data, item_data):
    """生成用户偏好摘要"""
    user_reviews = [r for r in review_data.values() if r['user_id'] == user_id]
    if not user_reviews:
        return "No historical reviews found"
    
    # 评分分布统计
    rating_dist = {1:0, 2:0, 3:0, 4:0, 5:0}
    category_dist = {}
    
    for review in user_reviews:
        # 处理评分
        stars = int(float(review.get('stars', 0)))
        if 1 <= stars <= 5:
            rating_dist[stars] += 1
        
        # 处理分类
        item = item_data.get(review['item_id'], {})
        categories = []
        if item.get('source') == 'yelp':
            categories = item.get('categories', '').split(', ')
        elif item.get('source') == 'amazon':
            categories = item.get('categories', [])
        elif item.get('source') == 'goodreads':
            categories = [shelf['name'] for shelf in item.get('popular_shelves', [])[:3]]
        
        for cat in categories:
            category_dist[cat] = category_dist.get(cat, 0) + 1
    
    # 生成摘要文本
    total_ratings = sum(rating_dist.values())
    avg_rating = sum(k*v for k,v in rating_dist.items())/total_ratings if total_ratings else 0
    top_categories = sorted(category_dist.items(), key=lambda x: x[1], reverse=True)[:3]
    
    return f"""
    User Preference Summary:
    - Average Rating: {avg_rating:.1f}/5
    - Rating Distribution: {dict(rating_dist)}
    - Top Categories: {[f'{k}({v})' for k, v in top_categories]}
    - Total Reviews: {len(user_reviews)}
    """

class OptimizedRecommendationAgent(RecommendationAgent):
    """
    优化后的推荐Agent实现
    """
    def __init__(self, llm: LLMBase):
        super().__init__(llm=llm)
        self.max_token_limit = 12000  # 控制上下文长度

    def _format_item_info(self, item_info):
        """格式化单个物品信息"""
        lines = [
            f"Item ID: {item_info['item_id']}",
            f"Type: {item_info['type'].title()}",
            f"Title: {item_info.get('name') or item_info.get('title')}",
            f"Rating: {item_info.get('rating', 'N/A')}",
            f"Reviews: {item_info.get('review_count', 0)}",
            f"Features: {item_info.get('features', 'N/A')}"
        ]
        return '\n'.join(lines)

    def workflow(self):
        """核心工作流程"""
        # 1. 生成用户画像
        user_profile = generate_user_profile(
            self.task['user_id'],
            self.simulator.review_data,
            self.simulator.item_data
        )
        
        # 2. 处理候选物品
        candidate_items = []
        for item_id in self.task['candidate_list']:
            item = self.simulator.item_data.get(item_id, {})
            item_info = get_structured_item_info(item)
            candidate_items.append(self._format_item_info(item_info))
        
        # 3. 构建prompt
        prompt_template = """[Role]
You are an expert recommendation system. Rank the following 20 items based on user preferences:

[User Profile]
{user_profile}

[Candidate Items]
{items}

[Instructions]
1. Consider rating, review count and feature compatibility
2. Prioritize items matching user's preferred categories
3. Output MUST be exactly 20 item IDs in the format: ['id1','id2',...]

[Output]"""
        full_prompt = prompt_template.format(
            user_profile=user_profile,
            items='\n\n'.join(candidate_items)
        )
        
        # 4. 上下文长度控制
        current_tokens = num_tokens_from_string(full_prompt)
        if current_tokens > self.max_token_limit:
            logging.warning(f"Prompt too long ({current_tokens}t), truncating...")
            available_space = self.max_token_limit - num_tokens_from_string(prompt_template)
            candidate_text = '\n\n'.join(candidate_items)
            candidate_tokens = num_tokens_from_string(candidate_text)
            
            if candidate_tokens > available_space:
                keep_ratio = available_space / candidate_tokens
                keep_items = int(len(candidate_items) * keep_ratio)
                candidate_items = candidate_items[:max(keep_items,5)]
                
            full_prompt = prompt_template.format(
                user_profile=user_profile,
                items='\n\n'.join(candidate_items)
            )

        # 5. 调用LLM并解析结果
        result = self.llm(
            messages=[{"role": "user", "content": full_prompt}],
            temperature=0.3,
            max_tokens=500
        )
        
        return self._parse_output(result)

    def _parse_output(self, text):
        """改进的结果解析方法"""
        try:
            # 尝试多种匹配方式
            matches = re.findall(r"'([A-Za-z0-9-]+)'", text)
            if len(matches) >= 20:
                return matches[:20]
            
            # 保底方案：返回原始顺序
            logging.warning("Failed to parse valid list, returning default order")
            return self.task['candidate_list']
        except Exception as e:
            logging.error(f"Parsing error: {str(e)}")
            return self.task['candidate_list']