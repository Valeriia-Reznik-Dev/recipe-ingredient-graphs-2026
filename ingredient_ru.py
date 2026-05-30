"""Простой словарь EN → RU для частых ингредиентов RecipeNLG."""

ING_RU = {
    'salt': 'соль', 'sugar': 'сахар', 'flour': 'мука', 'eggs': 'яйца', 'egg': 'яйцо',
    'butter': 'сливочное масло', 'onion': 'лук', 'onions': 'лук', 'milk': 'молоко',
    'vanilla': 'ваниль', 'water': 'вода', 'margarine': 'маргарин', 'pepper': 'перец',
    'black pepper': 'чёрный перец', 'brown sugar': 'коричневый сахар',
    'baking powder': 'разрыхлитель', 'baking soda': 'сода', 'cinnamon': 'корица',
    'nuts': 'орехи', 'cream cheese': 'сливочный сыр', 'celery': 'сельдерей',
    'sour cream': 'сметана', 'pineapple': 'ананас', 'pecans': 'пекан',
    'tomatoes': 'помидоры', 'tomato': 'помидор', 'cheddar cheese': 'сыр чеддер',
    'oil': 'масло', 'olive oil': 'оливковое масло', 'vegetable oil': 'растительное масло',
    'lemon juice': 'лимонный сок', 'garlic': 'чеснок', 'mayonnaise': 'майонез',
    'vinegar': 'уксус', 'green pepper': 'зелёный перец', 'chicken': 'курица',
    'ground beef': 'фарш говяжий', 'beef': 'говядина', 'pork': 'свинина',
    'shortening': 'кулинарный жир', 'potatoes': 'картофель', 'potato': 'картофель',
    'carrots': 'морковь', 'carrot': 'морковь', 'mushrooms': 'грибы',
    'powdered sugar': 'сахарная пудра', 'honey': 'мёд', 'rice': 'рис',
    'pasta': 'паста', 'bread': 'хлеб', 'cheese': 'сыр', 'cream': 'сливки',
    'yogurt': 'йогурт', 'basil': 'базилик', 'oregano': 'орегано',
    'parsley': 'петрушка', 'ginger': 'имбирь', 'soy sauce': 'соевый соус',
    'mustard': 'горчица', 'ketchup': 'кетчуп', 'tomato sauce': 'томатный соус',
    'chocolate': 'шоколад', 'cocoa': 'какао', 'apple': 'яблоко', 'apples': 'яблоки',
    'banana': 'банан', 'bananas': 'бананы', 'strawberries': 'клубника',
    'blueberries': 'черника', 'corn': 'кукуруза', 'cornstarch': 'крахмал',
    'flour tortillas': 'мука (тортильи)', 'tortilla': 'тортилья',
    'sausage': 'колбаса', 'bacon': 'бекон', 'ham': 'ветчина', 'turkey': 'индейка',
    'shrimp': 'креветки', 'salmon': 'лосось', 'tuna': 'тунец',
    'lentils': 'чечевица', 'beans': 'фасоль', 'chickpeas': 'нут',
    'tofu': 'тофу', 'spinach': 'шпинат', 'broccoli': 'брокколи',
    'zucchini': 'кабачок', 'cucumber': 'огурец', 'lettuce': 'салат',
    'avocado': 'авокадо', 'coconut': 'кокос', 'coconut milk': 'кокосовое молоко',
    'almonds': 'миндаль', 'walnuts': 'грецкие орехи', 'peanut butter': 'арахисовая паста',
    'maple syrup': 'кленовый сироп', 'whipped cream': 'взбитые сливки',
    'evaporated milk': 'сгущённое молоко', 'condensed milk': 'сгущёнка',
    'yeast': 'дрожжи', 'gelatin': 'желатин', 'paprika': 'паприка',
    'chili powder': 'перец чили', 'cumin': 'зира', 'nutmeg': 'мускатный орех',
    'cloves': 'гвоздика', 'thyme': 'тимьян', 'rosemary': 'розмарин',
    'bay leaf': 'лавровый лист', 'wine': 'вино', 'beer': 'пиво',
    'chicken broth': 'куриный бульон', 'beef broth': 'говяжий бульон',
    'stock': 'бульон', 'worcestershire sauce': 'соус Worcestershire',
    'cocoa powder': 'какао-порошок', 'jell-o': 'желе Jell-O', 'gelatin dessert': 'желе',
    'lumpia wrappers': 'обёртки для лумпии', 'spring roll wrappers': 'обёртки для спринг-роллов',
    'pistachio': 'фисташки', 'pistachios': 'фисташки', 'instant pudding': 'быстрорастворимый пудинг',
    'pudding': 'пудинг', 'light': 'лёгкий (light)', 'low-fat milk': 'обезжиренное молоко',
    'skim milk': 'обезжиренное молоко', 'sour lowfat milk': 'обезжиренная кисломолочка',
    'cream topping': 'взбитые сливки (топпинг)', 'pepitas': 'семечки тыквы',
    'hazelnut liqueur': 'ликёр с фундуком', 'crisp rice': 'хрустящий рис',
    'paraffin': 'парафин', 'silken': 'шёлковый тофу', 'lumpia': 'лумпия',
    'mirin': 'мирин', 'miso': 'мисо', 'dashi': 'даши', 'nori': 'нори',
    'wasabi': 'васаби', 'sake': 'саке', 'sesame oil': 'кунжутное масло',
    'rice vinegar': 'рисовый уксус', 'hoisin sauce': 'соус хойсин',
    'oyster sauce': 'устричный соус', 'green onions': 'зелёный лук',
    'sesame seeds': 'кунжут', 'garam masala': 'гарам масала', 'turmeric': 'куркума',
    'ghee': 'ги', 'cardamom': 'кардамон', 'coriander': 'кориандр',
    'dijon mustard': 'дижонская горчица', 'shallots': 'лук-шалот',
    'tarragon': 'тархун', 'gruyere': 'грюйер', 'creme fraiche': 'крем-фреш',
    'jalapeno': 'халапеньо', 'cilantro': 'кинза', 'black beans': 'чёрная фасоль',
    'salsa': 'сальса', 'lime': 'лайм', 'monterey jack': 'сыр Monterey Jack',
    'corn tortillas': 'кукурузные тортильи', 'barbecue sauce': 'соус барбекю',
    'mozzarella': 'моцарелла', 'parmesan': 'пармезан', 'ricotta': 'рикотта',
    'prosciutto': 'прошутто', 'marinara': 'соус маринара',
    'balsamic vinegar': 'бальзамический уксус',
    'vanilla extract': 'ванильный экстракт', 'all-purpose flour': 'мука общего назначения',
    'self-rising flour': 'мука с разрыхлителем', 'heavy cream': 'жирные сливки',
    'half-and-half': 'сливки 10–18%', 'buttermilk': 'пахта', 'molasses': 'патока',
    'raisins': 'изюм', 'oats': 'овсянка', 'oatmeal': 'овсянка',
    'orange soda': 'апельсиновая газировка', 'grain rice': 'зерновой рис',
    'lean ground meat': 'постный фарш', 'ground meat': 'фарш',
    'lean ground beef': 'постный говяжий фарш', 'thin noodles': 'тонкая лапша',
    'noodles': 'лапша', 'egg noodles': 'яичная лапша', 'spaghetti': 'спагетти',
    'macaroni': 'макароны', 'parmesan cheese': 'пармезан', 'mozzarella cheese': 'моцарелла',
    'cream of chicken soup': 'крем-суп с курицей', 'cream of mushroom soup': 'крем-суп с грибами',
    'chocolate chips': 'шоколадные капли', 'marshmallows': 'зефир', 'bread crumbs': 'панировка',
    'chicken breasts': 'куриная грудка', 'garlic powder': 'чесночный порошок',
    'white sugar': 'белый сахар', 'orange juice': 'апельсиновый сок', 'hamburger': 'говяжий фарш',
    'bell pepper': 'болгарский перец', 'cottage cheese': 'творог', 'tomato paste': 'томатная паста',
    'whipping cream': 'сливки для взбивания', 'boiling water': 'кипяток', 'cold water': 'холодная вода',
    'soda': 'газировка', 'oleo': 'маргарин', 'catsup': 'кетчуп', 'crisco': 'Crisco',
    'lemonade mix': 'смесь для лимонада', 'coconut instant pudding': 'кокосовый быстрый пудинг',
    'savoy': 'капуста савойская', 'shredded cheese': 'тёртый сыр', 'tortilla wraps': 'тортильи-обёртки',
    'lean lamb': 'постная баранина', 'graham cracker crumbs': 'крошка из крекеров Graham',
    'graham crackers': 'крекеры Graham', 'graham-crackers': 'крекеры Graham',
    'beef tips': 'кусочки говядины', 'sharp cheddar': 'острый чеддер',
    'sharp cheddar cheese': 'острый сыр чеддер', 'shredded sharp cheddar': 'тёртый острый чеддер',
    'lemon instant pudding': 'лимонный пудинг быстрого приготовления',
    'golden butter': 'золотистое сливочное масло', 'chicken breasts halves': 'половинки куриной грудки',
    'cherry tomatoes': 'помидоры черри', 'halved cherry tomatoes': 'половинки помидоров черри',
    'store-bought': 'магазинный продукт', 'ruby port': 'рубиновый портвейн', 'bone': 'кость',
    'chop suey vegetables': 'овощи для chop suey', 'white chocolate': 'белый шоколад',
    "s white chocolate": 'белый шоколад', 'egg whites': 'белки яиц', 'egg yolks': 'желтки яиц',
    'cabbage': 'капуста', 'pumpkin': 'тыква', 'crackers': 'крекеры', 'tomato soup': 'томатный суп',
    'allspice': 'душистый перец', 'pineapple juice': 'ананасовый сок',
    'yellow cake mix': 'смесь для жёлтого торта', 'cream of tartar': 'винный камень',
    'lamb': 'баранина', 'port': 'портвейн',     'preparation': 'заготовка',
    'stone-ground mustard': 'зернистая горчица', 'stone ground mustard': 'зернистая горчица',
    'blueberry yogurt': 'черничный йогурт',
    'coriander seeds': 'семена кориандра', 'whole coriander seeds': 'цельные семена кориандра',
    'quartered cherry tomatoes': 'четвертинки помидоров черри',
    'tomatillos': 'томатильо', 'fresh tomatillos': 'свежие томатильо',
    'mashed potatoes': 'картофельное пюре', 'instant mashed potatoes': 'пюре быстрого приготовления',
    'dry mustard': 'сухая горчица', 'ground dry mustard': 'молотая сухая горчица',
    'clove garlic': 'зубчик чеснока', 'confectioners sugar': 'сахарная пудра',
    'garlic salt': 'чесночная соль', 'salad oil': 'масло для салата',
    'swiss cheese': 'сыр швейцарский', 'peanuts': 'арахис', 'green peppers': 'зелёный перец',
    'cooking oil': 'растительное масло', 'ground cinnamon': 'молотая корица',
    'lemon': 'лимон', 'peaches': 'персики', 'olives': 'оливки', 'olive': 'оливки',
    'cucumbers': 'огурцы', 'kidney beans': 'красная фасоль', 'cauliflower': 'цветная капуста',
    'ginger ale': 'имбирный эль', 'tomato juice': 'томатный сок', 'oranges': 'апельсины',
    'black olives': 'чёрные оливки', 'cake flour': 'мука для выпечки',
    'cayenne pepper': 'перец кайенский', 'curry powder': 'карри', 'pork chops': 'свиные отбивные',
    'sweet potatoes': 'батат', 'peas': 'горох', 'syrup': 'сироп', 'orange': 'апельсин',
    'cloves': 'гвоздика', 'ground cloves': 'молотая гвоздика',
    'stew meat': 'мясо для тушения', 'stew': 'тушёное блюдо', 'roast': 'жаркое',
    'chili': 'чили', 'sauce': 'соус', 'soup mix': 'суповая смесь', 'onion soup mix': 'смесь для лукового супа',
    'onion soup': 'луковый суп', 'dip mix': 'смесь для дипа', 'dressing mix': 'смесь для заправки',
    'ranch dressing mix': 'смесь ranch', 'taco seasoning': 'приправа для тако',
    'italian seasoning': 'итальянские травы', 'poultry seasoning': 'приправа для птицы',
    'pie filling': 'начинка для пирога', 'apple pie filling': 'яблочная начинка',
    'cherry pie filling': 'вишнёвая начинка', 'pie crust': 'корж для пирога',
    'refrigerated pie crusts': 'охлаждённые коржи', 'crescent rolls': 'рогалики из слоёного теста',
    'refrigerated crescent rolls': 'охлаждённые рогалики', 'bisquick': 'Bisquick',
    'self rising flour': 'мука с разрыхлителем', 'unsalted butter': 'несолёное масло',
    'salted butter': 'солёное масло', 'melted butter': 'растопленное масло',
    'softened butter': 'размягчённое масло', 'stick butter': 'пачка масла',
    'egg beaters': 'яичный заменитель', 'nonfat dry milk': 'сухое обезжиренное молоко',
    'dry milk': 'сухое молоко', 'evaporated milk': 'сгущённое молоко',
    'sweetened condensed milk': 'сгущённое молоко с сахаром', 'corn syrup': 'кукурузный сироп',
    'light corn syrup': 'лёгкий кукурузный сироп', 'dark corn syrup': 'тёмный кукурузный сироп',
    'vanilla pudding': 'ванильный пудинг', 'chocolate pudding': 'шоколадный пудинг',
    'butterscotch chips': 'карамельные капли', 'butterscotch morsels': 'карамельные капли',
    'semi sweet chocolate chips': 'горькие шоколадные капли', 'semi-sweet chocolate chips': 'горькие шоколадные капли',
    'mini marshmallows': 'мини-зефир', 'miniature marshmallows': 'мини-зефир',
    'cool whip': 'Cool Whip', 'frozen whipped topping': 'замороженные взбитые сливки',
    'whipped topping': 'взбитый топпинг', 'red food coloring': 'красный пищевой краситель',
    'green food coloring': 'зелёный пищевой краситель', 'food coloring': 'пищевой краситель',
    'active dry yeast': 'сухие активные дрожжи', 'fast rising yeast': 'быстрые дрожжи',
    'pie apples': 'яблоки для пирога', 'apple pie spice': 'пряности для яблочного пира',
    'pumpkin pie spice': 'пряности для тыквенного пирога', 'apple cider vinegar': 'яблочный уксус',
    'red wine vinegar': 'красный винный уксус', 'white wine vinegar': 'белый винный уксус',
    'apple cider': 'яблочный сидр', 'cranberry sauce': 'клюквенный соус',
    'cranberries': 'клюква', 'cranberry': 'клюква', 'blueberry': 'черника',
    'strawberry': 'клубника', 'raspberry': 'малина', 'peach': 'персик',
    'apricot': 'абрикос', 'plum': 'слива', 'cherry': 'вишня', 'cherries': 'вишня',
    'zucchini': 'кабачок', 'squash': 'тыква', 'yellow squash': 'жёлтый кабачок',
    'green beans': 'стручковая фасоль', 'string beans': 'стручковая фасоль',
    'lima beans': 'лима', 'navy beans': 'белая фасоль', 'pinto beans': 'пинто',
    'refried beans': 'фасоль refried', 'black eyed peas': 'чёрноглазый горох',
    'split peas': 'колотый горох', 'green peas': 'зелёный горох',
    'bell peppers': 'болгарский перец', 'red bell pepper': 'красный болгарский перец',
    'green bell pepper': 'зелёный болгарский перец', 'yellow bell pepper': 'жёлтый болгарский перец',
    'jalapeno peppers': 'перец халапеньо', 'green chilies': 'зелёный чили',
    'green chilis': 'зелёный чили', 'diced green chilies': 'рубленый зелёный чили',
    'green chili peppers': 'зелёный перец чили', 'chilies': 'перец чили',
    'chili peppers': 'перец чили', 'hot pepper sauce': 'острый перечный соус',
    'hot sauce': 'острый соус', 'steak sauce': 'соус для стейка', 'soy': 'соевый',
    'fish sauce': 'рыбный соус', 'teriyaki sauce': 'соус терияки', 'sweet and sour sauce': 'кисло-сладкий соус',
    'spaghetti sauce': 'соус для спагетти', 'pizza sauce': 'соус для пиццы',
    'alfredo sauce': 'соус альфредо', 'enchilada sauce': 'соус энchilada',
    'salsa verde': 'зелёная сальса', 'picante sauce': 'острая сальса',
    'rotel tomatoes': 'помидоры Rotel', 'diced tomatoes': 'рубленые помидоры',
    'crushed tomatoes': 'тёртые помидоры', 'stewed tomatoes': 'тушёные помидоры',
    'whole tomatoes': 'цельные помидоры', 'tomato puree': 'томатное пюре',
    'tomato catsup': 'кетчуп', 'tomato ketchup': 'кетчуп', 'tomato': 'помидор',
    'mushroom soup': 'грибной суп', 'cream soup': 'крем-суп', 'celery soup': 'суп из сельдерея',
    'celery seed': 'семена сельдерея', 'celery salt': 'сельдерейная соль',
    'onion powder': 'луковый порошок', 'onion flakes': 'лук сушёный',
    'dried onion': 'сушёный лук', 'dried minced onion': 'сушёный рубленый лук',
    'green onion': 'зелёный лук', 'scallions': 'зелёный лук', 'leeks': 'лук-порей',
    'shallot': 'лук-шалот', 'garlic cloves': 'зубчики чеснока', 'minced garlic': 'рубленый чеснок',
    'roasted garlic': 'жареный чеснок', 'ginger root': 'корень имбиря',
    'ground ginger': 'молотый имбирь', 'ground nutmeg': 'молотый мускатный орех',
    'ground cumin': 'молотая зира', 'ground coriander': 'молотый кориандр',
    'ground pepper': 'молотый перец', 'white pepper': 'белый перец',
    'red pepper': 'красный перец', 'red pepper flakes': 'хлопья красного перца',
    'crushed red pepper': 'хлопья красного перца', 'crushed red pepper flakes': 'хлопья красного перца',
    'pepper flakes': 'перечные хлопья', 'seasoned salt': 'соль с приправами',
    'garlic pepper': 'чесночный перец', 'lemon pepper': 'лимонный перец',
    'meat': 'мясо', 'tips': 'кусочки', 'cubes': 'кубики', 'strips': 'полоски',
    'slices': 'ломтики', 'pieces': 'кусочки', 'halves': 'половинки', 'half': 'половина',
    'breast': 'грудка', 'breasts': 'грудка', 'thighs': 'бёдра', 'wings': 'крылья',
    'drumsticks': 'голени', 'tenderloin': 'вырезка', 'sirloin': 'sirloin',
    'round steak': 'стейк round', 'steak': 'стейк', 'roast': 'жаркое',
    'chuck roast': 'чак рост', 'pot roast': 'пот-роуст', 'corned beef': 'солонина',
    'pastrami': 'пастрами', 'salami': 'салями', 'pepperoni': 'пепперони',
    'hot dogs': 'сосиски', 'frankfurters': 'сосиски', 'smoked sausage': 'копчёная колбаса',
    'italian sausage': 'итальянская колбаса', 'breakfast sausage': 'колбаса для завтрака',
    'ground pork': 'свиной фарш', 'ground turkey': 'фарш индейки', 'ground chicken': 'куриный фарш',
    'ground lamb': 'бараний фарш', 'boneless': 'без кости', 'skinless': 'без кожи',
    'bone in': 'на кости', 'trimmed': 'очищенный', 'boneless skinless chicken breasts': 'куриная грудка без кости и кожи',
}

_SKIP_WORDS = frozenset({'of', 'for', 'and', 'with', 'a', 'an', 'the', 'in', 'on', 'to', 'or', 'from', 'at', 'by'})
_RUNTIME_RU: dict[str, str] = {}

_MODIFIERS = {
    'lean': 'постный', 'extra': 'extra', 'light': 'лёгкий', 'low-fat': 'обезжиренный',
    'fresh': 'свежий', 'dried': 'сушёный', 'frozen': 'замороженный',
    'chopped': 'рубленый', 'sliced': 'нарезанный', 'grated': 'тёртый',
    'shredded': 'тёртый', 'cooked': 'варёный', 'raw': 'сырой',
    'white': 'белый', 'brown': 'коричневый', 'red': 'красный', 'green': 'зелёный',
    'yellow': 'жёлтый', 'black': 'чёрный', 'thin': 'тонкий', 'thick': 'толстый',
    'halved': 'половинки', 'golden': 'золотистый', 'store-bought': 'магазинный',
    'blueberry': 'черничный', 'strawberry': 'клубничный', 'raspberry': 'малиновый',
    'stone-ground': 'зернистая', 'whole': 'цельные', 'quartered': 'четвертинки',
    'instant': 'быстрого приготовления', 'dry': 'сухая', 'ground': 'молотый',
    'lemon': 'лимонный', 'sharp': 'острый', 'unsalted': 'несолёное', 'salted': 'солёное',
    'melted': 'растопленное', 'softened': 'размягчённое', 'refrigerated': 'охлаждённое',
    'frozen': 'замороженное', 'canned': 'консервированное', 'undrained': 'не слитое',
    'drained': 'слитое', 'diced': 'рубленое', 'minced': 'мелко нарезанное',
    'crushed': 'тёртое', 'whole': 'цельное', 'nonfat': 'обезжиренное', 'fat free': 'обезжиренное',
    'low sodium': 'малосолёное', 'no salt added': 'без добавленной соли',
    'sweet': 'сладкое', 'sour': 'кислое', 'hot': 'острое', 'mild': 'неострое',
    'large': 'крупное', 'small': 'мелкое', 'medium': 'среднее', 'boneless': 'без кости',
    'skinless': 'без кожи', 'lean': 'постное', 'dark': 'тёмное', 'light': 'лёгкое',
}


def _normalize(name) -> str:
    key = str(name).strip().lower()
    return key.strip("'\"").replace('-', ' ')


def _dict_lookup(key: str) -> str | None:
    for variant in (key, key.replace(' ', '-'), key.replace('-', ' ')):
        if variant in ING_RU:
            return ING_RU[variant]
    return None


def _prefix_tail_translate(key: str) -> str | None:
    parts = key.split()
    for i in range(1, len(parts)):
        tail = ' '.join(parts[i:])
        tail_ru = _dict_lookup(tail)
        if tail_ru is None:
            continue
        head = parts[:i]
        if head and all(h in _MODIFIERS for h in head):
            prefix = ' '.join(_MODIFIERS[h] for h in head)
            return f'{prefix} {tail_ru}'
    return None


def _decompose_translate(key: str) -> str | None:
    parts = key.split()
    out: list[str] = []
    i = 0
    translated_any = False
    while i < len(parts):
        matched = False
        for j in range(len(parts), i, -1):
            chunk = ' '.join(parts[i:j])
            if chunk in _SKIP_WORDS:
                i = j
                matched = True
                break
            chunk_ru = _dict_lookup(chunk)
            if chunk_ru is not None:
                out.append(chunk_ru)
                i = j
                matched = True
                translated_any = True
                break
            if chunk in _MODIFIERS:
                out.append(_MODIFIERS[chunk])
                i = j
                matched = True
                translated_any = True
                break
        if not matched:
            out.append(parts[i])
            i += 1
    if not translated_any:
        return None
    result = ' '.join(out)
    if result == key:
        return None
    return result


def _translate_key(key: str) -> str:
    direct = _dict_lookup(key)
    if direct is not None:
        return direct
    prefixed = _prefix_tail_translate(key)
    if prefixed is not None:
        return prefixed
    decomposed = _decompose_translate(key)
    if decomposed is not None:
        return decomposed
    return key


def preload_pool(names) -> int:
    """Предперевести пул ингредиентов (вызывать после загрузки графа)."""
    new = 0
    for name in names:
        key = _normalize(name)
        if key in _RUNTIME_RU:
            continue
        tr = _translate_key(key)
        _RUNTIME_RU[key] = tr
        if tr != key:
            new += 1
    return new


def ru(name: str) -> str:
    """Вернуть русское название; если нет в словаре — разбор по частям или исходное имя."""
    key = _normalize(name)
    if key in _RUNTIME_RU:
        return _RUNTIME_RU[key]
    tr = _translate_key(key)
    _RUNTIME_RU[key] = tr
    return tr


def ru_list(names: list[str]) -> list[str]:
    return [ru(n) for n in names]
