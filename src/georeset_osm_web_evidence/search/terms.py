FRENCH_TERMS_BY_CATEGORY = {
    "forest": [
        "forêt",
        "biodiversité",
        "gestion forestière",
        "environnement",
    ],
    "agriculture": [
        "agriculture",
        "exploitation agricole",
        "culture",
        "terroir",
    ],
    "wetland": [
        "zone humide",
        "biodiversité",
        "conservation",
        "environnement",
    ],
    "protected_area": [
        "réserve naturelle",
        "protection",
        "biodiversité",
        "conservation",
    ],
    "default": [
        "environnement",
        "biodiversité",
        "site naturel",
    ],
}

ENGLISH_TERMS_BY_CATEGORY = {
    "forest": [
        "forest",
        "biodiversity",
        "forest management",
        "environment",
    ],
    "agriculture": [
        "agriculture",
        "farm",
        "crops",
        "land use",
    ],
    "wetland": [
        "wetland",
        "biodiversity",
        "conservation",
        "environment",
    ],
    "protected_area": [
        "nature reserve",
        "protection",
        "biodiversity",
        "conservation",
    ],
    "default": [
        "environment",
        "biodiversity",
        "natural site",
    ],
}

SPANISH_TERMS_BY_CATEGORY = {
    "forest": [
        "bosque",
        "biodiversidad",
        "gestión forestal",
        "medio ambiente",
    ],
    "agriculture": [
        "agricultura",
        "explotación agrícola",
        "cultivo",
        "uso del suelo",
    ],
    "wetland": [
        "humedal",
        "biodiversidad",
        "conservación",
        "medio ambiente",
    ],
    "protected_area": [
        "reserva natural",
        "protección",
        "biodiversidad",
        "conservación",
    ],
    "default": [
        "medio ambiente",
        "biodiversidad",
        "sitio natural",
    ],
}

PORTUGUESE_TERMS_BY_CATEGORY = {
    "forest": [
        "floresta",
        "biodiversidade",
        "gestão florestal",
        "meio ambiente",
    ],
    "agriculture": [
        "agricultura",
        "exploração agrícola",
        "cultivo",
        "uso do solo",
    ],
    "wetland": [
        "zona úmida",
        "biodiversidade",
        "conservação",
        "meio ambiente",
    ],
    "protected_area": [
        "reserva natural",
        "proteção",
        "biodiversidade",
        "conservação",
    ],
    "default": [
        "meio ambiente",
        "biodiversidade",
        "sítio natural",
    ],
}

ARABIC_TERMS_BY_CATEGORY = {
    "forest": [
        "غابة",
        "التنوع البيولوجي",
        "إدارة الغابات",
        "البيئة",
    ],
    "agriculture": [
        "الزراعة",
        "حقل زراعي",
        "محاصيل",
        "استخدام الأراضي",
    ],
    "wetland": [
        "أرض رطبة",
        "التنوع البيولوجي",
        "الحفاظ",
        "البيئة",
    ],
    "protected_area": [
        "محمية طبيعية",
        "الحماية",
        "التنوع البيولوجي",
        "الحفاظ",
    ],
    "default": [
        "البيئة",
        "التنوع البيولوجي",
        "موقع طبيعي",
    ],
}

SINHALA_TERMS_BY_CATEGORY = {
    "forest": [
        "වනාන්තරය",
        "ජෛව විවිධත්වය",
        "වනාන්තර කළමනාකරණය",
        "පරිසරය",
    ],
    "agriculture": [
        "කෘෂිකර්මය",
        "ගොවිපල",
        "වගාව",
        "ඉඩම් භාවිතය",
    ],
    "wetland": [
        "තෙත්බිම",
        "ජෛව විවිධත්වය",
        "සංරක්ෂණය",
        "පරිසරය",
    ],
    "protected_area": [
        "ස්වාභාවික රක්ෂිතය",
        "ආරක්ෂාව",
        "ජෛව විවිධත්වය",
        "සංරක්ෂණය",
    ],
    "default": [
        "පරිසරය",
        "ජෛව විවිධත්වය",
        "ස්වාභාවික ස්ථානය",
    ],
}


def category_terms(
    forest: str,
    biodiversity: str,
    forest_management: str,
    environment: str,
    agriculture: str,
    farm: str,
    crops: str,
    land_use: str,
    wetland: str,
    conservation: str,
    nature_reserve: str,
    protection: str,
    natural_site: str,
) -> dict[str, list[str]]:
    return {
        "forest": [
            forest,
            biodiversity,
            forest_management,
            environment,
        ],
        "agriculture": [
            agriculture,
            farm,
            crops,
            land_use,
        ],
        "wetland": [
            wetland,
            biodiversity,
            conservation,
            environment,
        ],
        "protected_area": [
            nature_reserve,
            protection,
            biodiversity,
            conservation,
        ],
        "default": [
            environment,
            biodiversity,
            natural_site,
        ],
    }


ADDITIONAL_TERMS_BY_LANGUAGE = {
    "am": category_terms("ደን", "ባዮሎጂያዊ ብዝሃነት", "የደን አስተዳደር", "አካባቢ", "ግብርና", "እርሻ", "ሰብሎች", "የመሬት አጠቃቀም", "እርጥብ መሬት", "ጥበቃ", "የተፈጥሮ መጠባበቂያ", "ጥበቃ", "የተፈጥሮ ቦታ"),
    "az": category_terms("meşə", "biomüxtəliflik", "meşə idarəçiliyi", "ətraf mühit", "kənd təsərrüfatı", "ferma", "əkinlər", "torpaq istifadəsi", "bataqlıq", "mühafizə", "təbiət qoruğu", "qorunma", "təbii ərazi"),
    "bg": category_terms("гора", "биоразнообразие", "управление на горите", "околна среда", "земеделие", "ферма", "култури", "земеползване", "влажна зона", "опазване", "природен резерват", "защита", "природен обект"),
    "bi": category_terms("forest", "biodiversity", "forest management", "environment", "agriculture", "farm", "crops", "land use", "wetland", "conservation", "nature reserve", "protection", "natural site"),
    "bn": category_terms("বন", "জীববৈচিত্র্য", "বন ব্যবস্থাপনা", "পরিবেশ", "কৃষি", "খামার", "ফসল", "ভূমি ব্যবহার", "জলাভূমি", "সংরক্ষণ", "প্রাকৃতিক সংরক্ষণ এলাকা", "সুরক্ষা", "প্রাকৃতিক স্থান"),
    "bs": category_terms("šuma", "biodiverzitet", "upravljanje šumama", "okoliš", "poljoprivreda", "farma", "usjevi", "korištenje zemljišta", "močvara", "očuvanje", "prirodni rezervat", "zaštita", "prirodno područje"),
    "ca": category_terms("bosc", "biodiversitat", "gestió forestal", "medi ambient", "agricultura", "explotació agrícola", "conreus", "ús del sòl", "zona humida", "conservació", "reserva natural", "protecció", "espai natural"),
    "cs": category_terms("les", "biologická rozmanitost", "lesní hospodaření", "životní prostředí", "zemědělství", "farma", "plodiny", "využití půdy", "mokřad", "ochrana přírody", "přírodní rezervace", "ochrana", "přírodní lokalita"),
    "da": category_terms("skov", "biodiversitet", "skovforvaltning", "miljø", "landbrug", "gård", "afgrøder", "arealanvendelse", "vådområde", "naturbeskyttelse", "naturreservat", "beskyttelse", "naturområde"),
    "de": category_terms("Wald", "Biodiversität", "Forstwirtschaft", "Umwelt", "Landwirtschaft", "Bauernhof", "Nutzpflanzen", "Landnutzung", "Feuchtgebiet", "Naturschutz", "Naturschutzgebiet", "Schutz", "Naturgebiet"),
    "el": category_terms("δάσος", "βιοποικιλότητα", "διαχείριση δασών", "περιβάλλον", "γεωργία", "αγρόκτημα", "καλλιέργειες", "χρήση γης", "υγρότοπος", "διατήρηση", "φυσικό καταφύγιο", "προστασία", "φυσική τοποθεσία"),
    "et": category_terms("mets", "elurikkus", "metsamajandus", "keskkond", "põllumajandus", "talu", "põllukultuurid", "maakasutus", "märgala", "looduskaitse", "looduskaitseala", "kaitse", "loodusala"),
    "fa": category_terms("جنگل", "تنوع زیستی", "مدیریت جنگل", "محیط زیست", "کشاورزی", "مزرعه", "محصولات کشاورزی", "کاربری زمین", "تالاب", "حفاظت", "ذخیره‌گاه طبیعی", "حفاظت", "مکان طبیعی"),
    "fi": category_terms("metsä", "luonnon monimuotoisuus", "metsänhoito", "ympäristö", "maatalous", "maatila", "viljelykasvit", "maankäyttö", "kosteikko", "suojelu", "luonnonsuojelualue", "suojelu", "luontokohde"),
    "he": category_terms("יער", "מגוון ביולוגי", "ניהול יערות", "סביבה", "חקלאות", "חווה", "גידולים", "שימוש בקרקע", "ביצה", "שימור", "שמורת טבע", "הגנה", "אתר טבע"),
    "hr": category_terms("šuma", "bioraznolikost", "gospodarenje šumama", "okoliš", "poljoprivreda", "farma", "usjevi", "korištenje zemljišta", "močvara", "očuvanje", "prirodni rezervat", "zaštita", "prirodno područje"),
    "hy": category_terms("անտառ", "կենսաբազմազանություն", "անտառի կառավարում", "շրջակա միջավայր", "գյուղատնտեսություն", "ֆերմա", "մշակաբույսեր", "հողօգտագործում", "խոնավ տարածք", "պահպանություն", "բնության արգելոց", "պաշտպանություն", "բնական վայր"),
    "id": category_terms("hutan", "keanekaragaman hayati", "pengelolaan hutan", "lingkungan", "pertanian", "lahan pertanian", "tanaman pangan", "penggunaan lahan", "lahan basah", "konservasi", "cagar alam", "perlindungan", "situs alam"),
    "is": category_terms("skógur", "líffræðilegur fjölbreytileiki", "skógarstjórnun", "umhverfi", "landbúnaður", "býli", "ræktun", "landnotkun", "votlendi", "verndun", "friðland", "vernd", "náttúrusvæði"),
    "it": category_terms("foresta", "biodiversità", "gestione forestale", "ambiente", "agricoltura", "azienda agricola", "colture", "uso del suolo", "zona umida", "conservazione", "riserva naturale", "protezione", "sito naturale"),
    "ja": category_terms("森林", "生物多様性", "森林管理", "環境", "農業", "農場", "作物", "土地利用", "湿地", "保全", "自然保護区", "保護", "自然地域"),
    "ka": category_terms("ტყე", "ბიომრავალფეროვნება", "ტყის მართვა", "გარემო", "სოფლის მეურნეობა", "ფერმა", "კულტურები", "მიწათსარგებლობა", "ჭაობი", "კონსერვაცია", "ბუნებრივი ნაკრძალი", "დაცვა", "ბუნებრივი ადგილი"),
    "kk": category_terms("орман", "биоәртүрлілік", "орманды басқару", "қоршаған орта", "ауыл шаруашылығы", "ферма", "дақылдар", "жер пайдалану", "батпақты жер", "сақтау", "табиғи қорық", "қорғау", "табиғи аймақ"),
    "km": category_terms("ព្រៃ", "ជីវចម្រុះ", "ការគ្រប់គ្រងព្រៃឈើ", "បរិស្ថាន", "កសិកម្ម", "កសិដ្ឋាន", "ដំណាំ", "ការប្រើប្រាស់ដី", "ដីសើម", "អភិរក្ស", "ដែនជម្រកធម្មជាតិ", "ការពារ", "តំបន់ធម្មជាតិ"),
    "ko": category_terms("숲", "생물다양성", "산림 관리", "환경", "농업", "농장", "작물", "토지 이용", "습지", "보전", "자연 보호구역", "보호", "자연 지역"),
    "ky": category_terms("токой", "биологиялык ар түрдүүлүк", "токойду башкаруу", "айлана-чөйрө", "айыл чарба", "ферма", "өсүмдүктөр", "жерди пайдалануу", "саздак жер", "сактоо", "жаратылыш коругу", "коргоо", "табигый жай"),
    "lb": category_terms("Bësch", "Biodiversitéit", "Bëschwirtschaft", "Ëmwelt", "Landwirtschaft", "Bauerenhaff", "Kulturen", "Landnotzung", "Fiichtgebitt", "Conservatioun", "Naturschutzgebitt", "Schutz", "Naturplaz"),
    "lo": category_terms("ປ່າໄມ້", "ຊີວະນານາພັນ", "ການຈັດການປ່າໄມ້", "ສິ່ງແວດລ້ອມ", "ກະສິກໍາ", "ຟາມ", "ພືດປູກ", "ການນໍາໃຊ້ທີ່ດິນ", "ພື້ນທີ່ຊຸ່ມນ້ໍາ", "ການອະນຸລັກ", "ເຂດສະຫງວນທໍາມະຊາດ", "ການປົກປ້ອງ", "ສະຖານທີ່ທໍາມະຊາດ"),
    "mg": category_terms("ala", "fahasamihafana biolojika", "fitantanana ala", "tontolo iainana", "fambolena", "toeram-pambolena", "voly", "fampiasana tany", "honahona", "fiarovana", "tahiry voajanahary", "fiarovana", "toerana voajanahary"),
    "mk": category_terms("шума", "биодиверзитет", "управување со шуми", "животна средина", "земјоделство", "фарма", "култури", "користење на земјиште", "мочуриште", "зачувување", "природен резерват", "заштита", "природно место"),
    "ml": category_terms("കാട്", "ജൈവവൈവിധ്യം", "വന പരിപാലനം", "പരിസ്ഥിതി", "കൃഷി", "ഫാം", "വിളകൾ", "ഭൂമിയുടെ ഉപയോഗം", "തണ്ണീർത്തടം", "സംരക്ഷണം", "പ്രകൃതി സംരക്ഷണ കേന്ദ്രം", "സംരക്ഷണം", "പ്രകൃതി സ്ഥലം"),
    "mn": category_terms("ой", "биологийн олон янз байдал", "ойн менежмент", "байгаль орчин", "хөдөө аж ахуй", "ферм", "тариалан", "газрын ашиглалт", "намгархаг газар", "хамгаалал", "байгалийн нөөц газар", "хамгаалалт", "байгалийн газар"),
    "ms": category_terms("hutan", "biodiversiti", "pengurusan hutan", "alam sekitar", "pertanian", "ladang", "tanaman", "penggunaan tanah", "tanah lembap", "pemuliharaan", "rizab alam semula jadi", "perlindungan", "tapak semula jadi"),
    "my": category_terms("သစ်တော", "ဇီဝမျိုးစုံမျိုးကွဲ", "သစ်တောစီမံခန့်ခွဲမှု", "ပတ်ဝန်းကျင်", "စိုက်ပျိုးရေး", "လယ်ယာ", "သီးနှံများ", "မြေအသုံးချမှု", "ရေစိုမြေ", "ထိန်းသိမ်းရေး", "သဘာဝထိန်းသိမ်းရေးနယ်မြေ", "ကာကွယ်ရေး", "သဘာဝနေရာ"),
    "ne": category_terms("वन", "जैविक विविधता", "वन व्यवस्थापन", "वातावरण", "कृषि", "फार्म", "बाली", "भूमि प्रयोग", "सिमसार", "संरक्षण", "प्राकृतिक आरक्ष", "सुरक्षा", "प्राकृतिक स्थल"),
    "nl": category_terms("bos", "biodiversiteit", "bosbeheer", "milieu", "landbouw", "boerderij", "gewassen", "landgebruik", "wetland", "natuurbehoud", "natuurreservaat", "bescherming", "natuurgebied"),
    "no": category_terms("skog", "biologisk mangfold", "skogforvaltning", "miljø", "jordbruk", "gård", "avlinger", "arealbruk", "våtmark", "bevaring", "naturreservat", "vern", "naturområde"),
    "pl": category_terms("las", "bioróżnorodność", "gospodarka leśna", "środowisko", "rolnictwo", "gospodarstwo", "uprawy", "użytkowanie gruntów", "teren podmokły", "ochrona", "rezerwat przyrody", "ochrona", "obszar naturalny"),
    "ro": category_terms("pădure", "biodiversitate", "management forestier", "mediu", "agricultură", "fermă", "culturi", "utilizarea terenului", "zonă umedă", "conservare", "rezervație naturală", "protecție", "sit natural"),
    "ru": category_terms("лес", "биоразнообразие", "управление лесами", "окружающая среда", "сельское хозяйство", "ферма", "культуры", "землепользование", "водно-болотные угодья", "сохранение", "природный заповедник", "охрана", "природный объект"),
    "rw": category_terms("ishyamba", "urusobe rw'ibinyabuzima", "imicungire y'amashyamba", "ibidukikije", "ubuhinzi", "umurima", "imyaka", "imikoreshereze y'ubutaka", "igishanga", "kubungabunga", "pariki y'ibidukikije", "kurinda", "ahantu nyaburanga"),
    "sk": category_terms("les", "biodiverzita", "lesné hospodárstvo", "životné prostredie", "poľnohospodárstvo", "farma", "plodiny", "využívanie pôdy", "mokraď", "ochrana prírody", "prírodná rezervácia", "ochrana", "prírodná lokalita"),
    "sl": category_terms("gozd", "biotska raznovrstnost", "gospodarjenje z gozdovi", "okolje", "kmetijstvo", "kmetija", "pridelki", "raba zemljišč", "mokrišče", "ohranjanje", "naravni rezervat", "varstvo", "naravno območje"),
    "sm": category_terms("vaomatua", "ese'esega o meaola", "pulega o vaomatua", "siosiomaga", "fa'ato'aga", "fa'ato'aga", "fa'ato'aga fua", "fa'aogaina o fanua", "eleele susu", "fa'asao", "nofoaga fa'asao fa'anatura", "puipuiga", "nofoaga fa'anatura"),
    "so": category_terms("kayn", "kala duwanaanshaha noolaha", "maamulka kaymaha", "deegaanka", "beeraha", "beer", "dalagyo", "isticmaalka dhulka", "dhul qoyan", "ilaalin", "keyd dabiici ah", "ilaalin", "goob dabiici ah"),
    "sq": category_terms("pyll", "biodiversitet", "menaxhim i pyjeve", "mjedis", "bujqësi", "fermë", "kultura bujqësore", "përdorim i tokës", "ligatinë", "ruajtje", "rezervat natyror", "mbrojtje", "zonë natyrore"),
    "sr": category_terms("шума", "биодиверзитет", "управљање шумама", "животна средина", "пољопривреда", "фарма", "усеви", "коришћење земљишта", "мочвара", "очување", "природни резерват", "заштита", "природно подручје"),
    "sv": category_terms("skog", "biologisk mångfald", "skogsförvaltning", "miljö", "jordbruk", "gård", "grödor", "markanvändning", "våtmark", "naturvård", "naturreservat", "skydd", "naturområde"),
    "sw": category_terms("msitu", "bioanuwai", "usimamizi wa misitu", "mazingira", "kilimo", "shamba", "mazao", "matumizi ya ardhi", "ardhioevu", "uhifadhi", "hifadhi ya asili", "ulinzi", "eneo la asili"),
    "tg": category_terms("ҷангал", "гуногунии биологӣ", "идоракунии ҷангал", "муҳити зист", "кишоварзӣ", "хоҷагӣ", "зироатҳо", "истифодаи замин", "ботлоқзор", "ҳифз", "мамнӯъгоҳи табиӣ", "муҳофизат", "макони табиӣ"),
    "th": category_terms("ป่า", "ความหลากหลายทางชีวภาพ", "การจัดการป่าไม้", "สิ่งแวดล้อม", "เกษตรกรรม", "ฟาร์ม", "พืชผล", "การใช้ที่ดิน", "พื้นที่ชุ่มน้ำ", "การอนุรักษ์", "เขตสงวนธรรมชาติ", "การคุ้มครอง", "พื้นที่ธรรมชาติ"),
    "tl": category_terms("gubat", "biodiversity", "pamamahala ng kagubatan", "kapaligiran", "agrikultura", "sakahan", "pananim", "paggamit ng lupa", "latian", "konserbasyon", "reserbang kalikasan", "proteksyon", "likas na lugar"),
    "tr": category_terms("orman", "biyolojik çeşitlilik", "orman yönetimi", "çevre", "tarım", "çiftlik", "ürünler", "arazi kullanımı", "sulak alan", "koruma", "doğa koruma alanı", "koruma", "doğal alan"),
    "ur": category_terms("جنگل", "حیاتیاتی تنوع", "جنگلات کا انتظام", "ماحول", "زراعت", "کھیت", "فصلیں", "زمین کا استعمال", "گیلی زمین", "تحفظ", "قدرتی محفوظ علاقہ", "حفاظت", "قدرتی مقام"),
    "vi": category_terms("rừng", "đa dạng sinh học", "quản lý rừng", "môi trường", "nông nghiệp", "trang trại", "cây trồng", "sử dụng đất", "đất ngập nước", "bảo tồn", "khu bảo tồn thiên nhiên", "bảo vệ", "địa điểm tự nhiên"),
    "zh": category_terms("森林", "生物多样性", "森林管理", "环境", "农业", "农场", "作物", "土地利用", "湿地", "保护", "自然保护区", "保护", "自然地点"),
}

TERMS_BY_LANGUAGE = {
    "fr": FRENCH_TERMS_BY_CATEGORY,
    "en": ENGLISH_TERMS_BY_CATEGORY,
    "es": SPANISH_TERMS_BY_CATEGORY,
    "pt": PORTUGUESE_TERMS_BY_CATEGORY,
    "ar": ARABIC_TERMS_BY_CATEGORY,
    "si": SINHALA_TERMS_BY_CATEGORY,
    **ADDITIONAL_TERMS_BY_LANGUAGE,
}
