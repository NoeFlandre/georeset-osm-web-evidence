import math

import pandas as pd
import geopandas as gpd

from georeset_osm_web_evidence.osm.geodataframe import WGS84_GEOD
from georeset_osm_web_evidence.osm.tags import ENVIRONMENTAL_TAGS


def compute_sample_size(
    target_sentences: int,
    planned_sentences_per_polygon: int,
) -> int:
    if planned_sentences_per_polygon <= 0:
        raise ValueError("planned_sentences_per_polygon must be positive")

    return -(-target_sentences // planned_sentences_per_polygon)


def bbox_config(
    bbox_id: str,
    bbox_label: str,
    country: str,
    world_region: str,
    local_language: str,
    bbox: tuple[float, float, float, float],
) -> dict:
    return {
        "bbox_id": bbox_id,
        "bbox_label": bbox_label,
        "country": country,
        "world_region": world_region,
        "local_language": local_language,
        "bbox": bbox,
    }


def _step_code(axis: str, step: int) -> str:
    direction = "p" if step > 0 else "m"

    return f"{axis}{direction}{abs(step)}"


def generate_bbox_expansions(
    anchor_bboxes: list[dict],
    expansion_radius_steps: int = 4,
    gap_factor: float = 1.5,
) -> list[dict]:
    expansions = []
    seen_bbox_ids = set()
    seen_coordinates = set()
    offset_steps = [
        (lat_step, lon_step)
        for lat_step in range(-expansion_radius_steps, expansion_radius_steps + 1)
        for lon_step in range(-expansion_radius_steps, expansion_radius_steps + 1)
        if not (lat_step == 0 and lon_step == 0)
    ]

    for lat_step, lon_step in offset_steps:
        for anchor in anchor_bboxes:
            south, west, north, east = anchor["bbox"]
            height = north - south
            width = east - west
            lat_spacing = height * gap_factor
            lon_spacing = width * gap_factor

            new_south = south + lat_step * lat_spacing
            new_north = new_south + height
            new_west = west + lon_step * lon_spacing
            new_east = new_west + width

            if new_south < -85 or new_north > 85:
                continue

            if new_west < -180 or new_east > 180:
                continue

            bbox_id = (
                f"{anchor['bbox_id']}_exp_"
                f"{_step_code('lat', lat_step)}_"
                f"{_step_code('lon', lon_step)}"
            )
            bbox = (
                round(new_south, 6),
                round(new_west, 6),
                round(new_north, 6),
                round(new_east, 6),
            )

            if bbox_id in seen_bbox_ids or bbox in seen_coordinates:
                continue

            seen_bbox_ids.add(bbox_id)
            seen_coordinates.add(bbox)
            expansions.append(
                bbox_config(
                    bbox_id=bbox_id,
                    bbox_label=f"{anchor['bbox_label']} expansion",
                    country=anchor["country"],
                    world_region=anchor["world_region"],
                    local_language=anchor["local_language"],
                    bbox=bbox,
                )
            )

    return expansions


WORLDWIDE_TRAINING_BBOXES = [
    bbox_config("fr_brittany", "Brittany wetlands", "France", "Europe", "fr", (47.8, -4.4, 48.2, -3.7)),
    bbox_config("es_catalonia", "Catalonia forests", "Spain", "Europe", "es", (41.6, 1.6, 42.0, 2.3)),
    bbox_config("it_tuscany", "Tuscany countryside", "Italy", "Europe", "it", (43.2, 10.6, 43.6, 11.3)),
    bbox_config("pt_north", "Northern Portugal", "Portugal", "Europe", "pt", (41.3, -8.5, 41.8, -7.8)),
    bbox_config("de_bavaria", "Bavarian forests", "Germany", "Europe", "de", (47.6, 11.0, 48.1, 11.8)),
    bbox_config("pl_bialowieza", "Bialowieza region", "Poland", "Europe", "pl", (52.5, 23.4, 52.9, 24.0)),
    bbox_config("se_smaland", "Smaland forests", "Sweden", "Europe", "sv", (56.6, 14.5, 57.1, 15.3)),
    bbox_config("no_oslofjord", "Oslofjord forests", "Norway", "Europe", "no", (59.3, 10.2, 59.8, 11.0)),
    bbox_config("gr_peloponnese", "Peloponnese uplands", "Greece", "Europe", "el", (37.4, 22.0, 37.9, 22.8)),
    bbox_config("ro_carpathians", "Romanian Carpathians", "Romania", "Europe", "ro", (45.2, 25.0, 45.7, 25.8)),
    bbox_config("uk_lake_district", "Lake District", "United Kingdom", "Europe", "en", (54.2, -3.4, 54.7, -2.6)),
    bbox_config("tr_black_sea", "Black Sea mountains", "Turkey", "Europe", "tr", (40.6, 39.0, 41.1, 39.8)),
    bbox_config("ma_atlas", "Middle Atlas", "Morocco", "Africa", "ar", (32.8, -5.4, 33.2, -4.7)),
    bbox_config("ke_central", "Central Kenya", "Kenya", "Africa", "sw", (-0.4, 36.7, 0.1, 37.4)),
    bbox_config("za_cape", "Western Cape", "South Africa", "Africa", "en", (-34.3, 18.2, -33.8, 19.0)),
    bbox_config("gh_ashanti", "Ashanti region", "Ghana", "Africa", "en", (6.3, -1.9, 6.8, -1.1)),
    bbox_config("et_highlands", "Ethiopian highlands", "Ethiopia", "Africa", "am", (9.0, 38.2, 9.5, 39.0)),
    bbox_config("tz_kilimanjaro", "Kilimanjaro region", "Tanzania", "Africa", "sw", (-3.4, 37.0, -2.9, 37.8)),
    bbox_config("ug_lake_victoria", "Lake Victoria north", "Uganda", "Africa", "en", (0.0, 32.0, 0.5, 32.8)),
    bbox_config("mg_east", "Eastern Madagascar", "Madagascar", "Africa", "mg", (-19.2, 48.0, -18.7, 48.8)),
    bbox_config("sn_casamance", "Casamance", "Senegal", "Africa", "fr", (12.4, -16.4, 12.9, -15.6)),
    bbox_config("rw_nyungwe", "Nyungwe region", "Rwanda", "Africa", "rw", (-2.7, 29.0, -2.2, 29.8)),
    bbox_config("cm_west", "Cameroon highlands", "Cameroon", "Africa", "fr", (5.3, 9.8, 5.8, 10.6)),
    bbox_config("na_highlands", "Namibian highlands", "Namibia", "Africa", "en", (-22.8, 16.6, -22.3, 17.4)),
    bbox_config("in_kerala", "Kerala western ghats", "India", "Asia", "ml", (10.0, 76.2, 10.5, 77.0)),
    bbox_config("jp_fuji", "Mount Fuji area", "Japan", "Asia", "ja", (35.2, 138.4, 35.6, 139.1)),
    bbox_config("id_bali", "Bali uplands", "Indonesia", "Asia", "id", (-8.6, 115.0, -8.1, 115.6)),
    bbox_config("th_chiang_mai", "Chiang Mai uplands", "Thailand", "Asia", "th", (18.5, 98.4, 19.0, 99.2)),
    bbox_config("vn_central", "Central Vietnam highlands", "Vietnam", "Asia", "vi", (12.0, 108.0, 12.5, 108.8)),
    bbox_config("ph_luzon", "Northern Luzon", "Philippines", "Asia", "tl", (16.3, 120.4, 16.8, 121.2)),
    bbox_config("np_hills", "Nepal middle hills", "Nepal", "Asia", "ne", (27.5, 84.8, 28.0, 85.6)),
    bbox_config("lk_central", "Sri Lanka central highlands", "Sri Lanka", "Asia", "si", (6.7, 80.5, 7.2, 81.3)),
    bbox_config("cn_yunnan", "Yunnan uplands", "China", "Asia", "zh", (24.6, 102.0, 25.1, 102.8)),
    bbox_config("kr_jeju", "Jeju island", "South Korea", "Asia", "ko", (33.2, 126.2, 33.7, 127.0)),
    bbox_config("my_sabah", "Sabah interior", "Malaysia", "Asia", "ms", (5.4, 116.0, 5.9, 116.8)),
    bbox_config("ge_caucasus", "Caucasus foothills", "Georgia", "Asia", "ka", (41.8, 43.8, 42.3, 44.6)),
    bbox_config("us_california", "Northern California", "United States", "North America", "en", (38.3, -123.2, 38.8, -122.4)),
    bbox_config("mx_oaxaca", "Oaxaca valleys", "Mexico", "North America", "es", (16.8, -97.1, 17.3, -96.4)),
    bbox_config("ca_quebec", "Southern Quebec", "Canada", "North America", "fr", (45.2, -72.8, 45.8, -71.9)),
    bbox_config("us_appalachia", "Southern Appalachians", "United States", "North America", "en", (35.4, -83.8, 35.9, -83.0)),
    bbox_config("us_vermont", "Vermont forests", "United States", "North America", "en", (43.8, -73.1, 44.3, -72.3)),
    bbox_config("us_oregon", "Oregon cascades", "United States", "North America", "en", (44.0, -122.7, 44.5, -121.9)),
    bbox_config("ca_bc", "British Columbia coast", "Canada", "North America", "en", (49.2, -123.7, 49.7, -122.9)),
    bbox_config("cr_central", "Costa Rica central", "Costa Rica", "North America", "es", (9.7, -84.3, 10.2, -83.5)),
    bbox_config("gt_highlands", "Guatemala highlands", "Guatemala", "North America", "es", (14.5, -91.0, 15.0, -90.2)),
    bbox_config("do_cordillera", "Dominican cordillera", "Dominican Republic", "North America", "es", (18.5, -71.2, 19.0, -70.4)),
    bbox_config("us_florida_wetlands", "Florida wetlands", "United States", "North America", "en", (25.6, -81.4, 26.1, -80.6)),
    bbox_config("us_colorado", "Colorado front range", "United States", "North America", "en", (39.2, -105.4, 39.7, -104.6)),
    bbox_config("br_minas", "Minas Gerais", "Brazil", "South America", "pt", (-20.4, -44.4, -19.8, -43.6)),
    bbox_config("cl_lakes", "Chile lake district", "Chile", "South America", "es", (-41.4, -73.2, -40.8, -72.3)),
    bbox_config("co_andes", "Colombian Andes", "Colombia", "South America", "es", (5.0, -75.9, 5.5, -75.1)),
    bbox_config("pe_cusco", "Cusco highlands", "Peru", "South America", "es", (-13.7, -72.2, -13.2, -71.4)),
    bbox_config("ar_misiones", "Misiones forest", "Argentina", "South America", "es", (-26.2, -54.9, -25.7, -54.1)),
    bbox_config("uy_canelones", "Canelones countryside", "Uruguay", "South America", "es", (-34.7, -56.5, -34.2, -55.7)),
    bbox_config("ec_andes", "Ecuadorian Andes", "Ecuador", "South America", "es", (-0.5, -78.9, 0.0, -78.1)),
    bbox_config("bo_yungas", "Bolivian Yungas", "Bolivia", "South America", "es", (-16.5, -68.2, -16.0, -67.4)),
    bbox_config("py_eastern", "Eastern Paraguay", "Paraguay", "South America", "es", (-25.7, -56.2, -25.2, -55.4)),
    bbox_config("ve_andes", "Venezuelan Andes", "Venezuela", "South America", "es", (8.4, -71.5, 8.9, -70.7)),
    bbox_config("br_santa_catarina", "Santa Catarina", "Brazil", "South America", "pt", (-27.8, -49.7, -27.3, -48.9)),
    bbox_config("ar_patagonia", "Northern Patagonia", "Argentina", "South America", "es", (-41.4, -71.7, -40.9, -70.9)),
    bbox_config("au_blue_mountains", "Blue Mountains", "Australia", "Oceania", "en", (-33.9, 150.1, -33.4, 150.7)),
    bbox_config("nz_waikato", "Waikato", "New Zealand", "Oceania", "en", (-38.2, 175.1, -37.6, 175.8)),
    bbox_config("au_tasmania", "Tasmania forests", "Australia", "Oceania", "en", (-42.8, 146.8, -42.3, 147.6)),
    bbox_config("au_victoria", "Victorian ranges", "Australia", "Oceania", "en", (-37.8, 145.4, -37.3, 146.2)),
    bbox_config("nz_canterbury", "Canterbury foothills", "New Zealand", "Oceania", "en", (-43.7, 171.3, -43.2, 172.1)),
    bbox_config("pg_highlands", "Papua New Guinea highlands", "Papua New Guinea", "Oceania", "en", (-6.2, 143.6, -5.7, 144.4)),
    bbox_config("fj_viti_levu", "Viti Levu", "Fiji", "Oceania", "en", (-18.2, 177.6, -17.7, 178.4)),
    bbox_config("au_queensland", "Queensland wet tropics", "Australia", "Oceania", "en", (-17.6, 145.4, -17.1, 146.2)),
]

BASE_WORLDWIDE_TRAINING_BBOX_COUNT = len(WORLDWIDE_TRAINING_BBOXES)

WORLDWIDE_TRAINING_BBOXES.extend([
    bbox_config("ru_karelia", "Karelian forests", "Russia", "Europe", "ru", (61.6, 31.0, 62.0, 31.8)),
    bbox_config("ru_moscow_oblast", "Moscow oblast forests", "Russia", "Europe", "ru", (55.3, 36.5, 55.8, 37.3)),
    bbox_config("ru_ural", "Ural foothills", "Russia", "Asia", "ru", (56.7, 59.2, 57.2, 60.0)),
    bbox_config("ru_altai", "Russian Altai", "Russia", "Asia", "ru", (51.6, 85.4, 52.1, 86.2)),
    bbox_config("ru_baikal", "Lake Baikal forests", "Russia", "Asia", "ru", (51.6, 104.8, 52.1, 105.6)),
    bbox_config("ru_kamchatka", "Kamchatka landscapes", "Russia", "Asia", "ru", (53.0, 158.2, 53.5, 159.0)),
    bbox_config("ru_primorye", "Primorye forests", "Russia", "Asia", "ru", (43.0, 132.0, 43.5, 132.8)),
    bbox_config("fi_lakeland", "Finnish Lakeland", "Finland", "Europe", "fi", (61.6, 25.0, 62.1, 25.8)),
    bbox_config("is_south", "Southern Iceland", "Iceland", "Europe", "is", (63.6, -20.8, 64.1, -20.0)),
    bbox_config("ch_alps", "Swiss Alps", "Switzerland", "Europe", "de", (46.4, 8.0, 46.9, 8.8)),
    bbox_config("at_tyrol", "Tyrolean Alps", "Austria", "Europe", "de", (47.0, 10.8, 47.5, 11.6)),
    bbox_config("nl_veluwe", "Veluwe forests", "Netherlands", "Europe", "nl", (52.0, 5.6, 52.5, 6.3)),
    bbox_config("be_ardennes", "Belgian Ardennes", "Belgium", "Europe", "fr", (50.0, 5.2, 50.5, 6.0)),
    bbox_config("cz_bohemia", "Bohemian countryside", "Czechia", "Europe", "cs", (49.7, 14.1, 50.2, 14.9)),
    bbox_config("sk_tatra", "Tatra foothills", "Slovakia", "Europe", "sk", (49.0, 19.2, 49.5, 20.0)),
    bbox_config("si_julian_alps", "Julian Alps", "Slovenia", "Europe", "sl", (46.1, 13.5, 46.6, 14.3)),
    bbox_config("hr_slavonia", "Slavonia countryside", "Croatia", "Europe", "hr", (45.2, 17.2, 45.7, 18.0)),
    bbox_config("bg_rila", "Rila mountains", "Bulgaria", "Europe", "bg", (42.0, 23.0, 42.5, 23.8)),
    bbox_config("ee_lahemaa", "Lahemaa region", "Estonia", "Europe", "et", (59.3, 25.6, 59.8, 26.4)),
    bbox_config("dz_kabylie", "Kabylie mountains", "Algeria", "Africa", "ar", (36.3, 4.0, 36.8, 4.8)),
    bbox_config("tn_dorsale", "Tunisian Dorsale", "Tunisia", "Africa", "ar", (36.1, 9.0, 36.6, 9.8)),
    bbox_config("ng_cross_river", "Cross River forests", "Nigeria", "Africa", "en", (5.4, 8.4, 5.9, 9.2)),
    bbox_config("ci_tai", "Tai forest region", "Cote d'Ivoire", "Africa", "fr", (5.6, -7.5, 6.1, -6.7)),
    bbox_config("ga_loango", "Loango region", "Gabon", "Africa", "fr", (-2.4, 9.2, -1.9, 10.0)),
    bbox_config("cd_kivu", "Kivu highlands", "DR Congo", "Africa", "fr", (-2.0, 28.6, -1.5, 29.4)),
    bbox_config("zm_copperbelt", "Zambian Copperbelt", "Zambia", "Africa", "en", (-13.1, 27.6, -12.6, 28.4)),
    bbox_config("zw_eastern_highlands", "Eastern Highlands", "Zimbabwe", "Africa", "en", (-19.0, 32.4, -18.5, 33.2)),
    bbox_config("mz_gorongosa", "Gorongosa region", "Mozambique", "Africa", "pt", (-18.9, 34.0, -18.4, 34.8)),
    bbox_config("bw_okavango", "Okavango Delta", "Botswana", "Africa", "en", (-19.6, 22.5, -19.1, 23.3)),
    bbox_config("mw_mulanje", "Mulanje region", "Malawi", "Africa", "en", (-16.2, 35.4, -15.7, 36.2)),
    bbox_config("ao_huila", "Huila highlands", "Angola", "Africa", "pt", (-14.9, 13.2, -14.4, 14.0)),
    bbox_config("kz_altai", "Kazakh Altai", "Kazakhstan", "Asia", "kk", (49.0, 84.6, 49.5, 85.4)),
    bbox_config("kg_tien_shan", "Tien Shan foothills", "Kyrgyzstan", "Asia", "ky", (42.4, 74.4, 42.9, 75.2)),
    bbox_config("mn_khangai", "Khangai mountains", "Mongolia", "Asia", "mn", (47.0, 101.0, 47.5, 101.8)),
    bbox_config("ir_caspian", "Caspian forests", "Iran", "Asia", "fa", (36.4, 51.8, 36.9, 52.6)),
    bbox_config("pk_himalaya", "Pakistan Himalayan foothills", "Pakistan", "Asia", "ur", (34.1, 73.0, 34.6, 73.8)),
    bbox_config("bd_sylhet", "Sylhet wetlands", "Bangladesh", "Asia", "bn", (24.5, 91.4, 25.0, 92.2)),
    bbox_config("mm_shan", "Shan hills", "Myanmar", "Asia", "my", (20.5, 96.5, 21.0, 97.3)),
    bbox_config("la_luang_prabang", "Luang Prabang uplands", "Laos", "Asia", "lo", (19.7, 102.0, 20.2, 102.8)),
    bbox_config("kh_cardamom", "Cardamom mountains", "Cambodia", "Asia", "km", (11.4, 103.2, 11.9, 104.0)),
    bbox_config("tw_central", "Central Taiwan", "Taiwan", "Asia", "zh", (23.6, 120.7, 24.1, 121.5)),
    bbox_config("us_maine", "Maine forests", "United States", "North America", "en", (44.6, -70.8, 45.1, -70.0)),
    bbox_config("us_minnesota", "Minnesota lakes", "United States", "North America", "en", (47.5, -92.6, 48.0, -91.8)),
    bbox_config("us_new_mexico", "New Mexico mountains", "United States", "North America", "en", (35.8, -106.0, 36.3, -105.2)),
    bbox_config("us_arizona", "Arizona uplands", "United States", "North America", "en", (34.7, -111.9, 35.2, -111.1)),
    bbox_config("ca_ontario", "Ontario shield", "Canada", "North America", "en", (45.0, -79.6, 45.5, -78.8)),
    bbox_config("ca_alberta", "Alberta foothills", "Canada", "North America", "en", (51.0, -115.4, 51.5, -114.6)),
    bbox_config("ca_yukon", "Yukon landscapes", "Canada", "North America", "en", (60.5, -135.5, 61.0, -134.7)),
    bbox_config("us_alaska", "Alaska panhandle", "United States", "North America", "en", (58.0, -135.0, 58.5, -134.2)),
    bbox_config("bz_maya", "Maya Mountains", "Belize", "North America", "en", (16.6, -89.1, 17.1, -88.3)),
    bbox_config("pa_darien", "Darien region", "Panama", "North America", "es", (8.0, -78.2, 8.5, -77.4)),
    bbox_config("br_amazon_para", "Para Amazon", "Brazil", "South America", "pt", (-3.0, -55.2, -2.5, -54.4)),
    bbox_config("br_parana", "Parana countryside", "Brazil", "South America", "pt", (-25.5, -50.6, -25.0, -49.8)),
    bbox_config("br_bahia", "Bahia landscapes", "Brazil", "South America", "pt", (-12.6, -41.4, -12.1, -40.6)),
    bbox_config("pe_amazon", "Peruvian Amazon", "Peru", "South America", "es", (-6.1, -76.2, -5.6, -75.4)),
    bbox_config("gy_highlands", "Guyana highlands", "Guyana", "South America", "en", (5.0, -59.6, 5.5, -58.8)),
    bbox_config("sr_brokopondo", "Brokopondo forests", "Suriname", "South America", "nl", (4.6, -55.4, 5.1, -54.6)),
    bbox_config("cl_central", "Central Chile", "Chile", "South America", "es", (-35.4, -72.2, -34.9, -71.4)),
    bbox_config("ar_cordoba", "Cordoba hills", "Argentina", "South America", "es", (-31.8, -64.8, -31.3, -64.0)),
    bbox_config("co_amazon", "Colombian Amazon", "Colombia", "South America", "es", (-1.2, -72.4, -0.7, -71.6)),
    bbox_config("gf_interior", "French Guiana interior", "French Guiana", "South America", "fr", (4.2, -53.2, 4.7, -52.4)),
    bbox_config("nz_otago", "Otago uplands", "New Zealand", "Oceania", "en", (-45.6, 169.5, -45.1, 170.3)),
    bbox_config("au_northern_territory", "Northern Territory savanna", "Australia", "Oceania", "en", (-13.5, 131.0, -13.0, 131.8)),
    bbox_config("au_south_west", "Western Australia forests", "Australia", "Oceania", "en", (-34.2, 116.0, -33.7, 116.8)),
    bbox_config("sb_guadalcanal", "Guadalcanal", "Solomon Islands", "Oceania", "en", (-9.8, 159.6, -9.3, 160.4)),
    bbox_config("vu_efate", "Efate island", "Vanuatu", "Oceania", "bi", (-17.9, 168.2, -17.4, 169.0)),
])

WORLDWIDE_EXPANSION_BBOXES = generate_bbox_expansions(
    WORLDWIDE_TRAINING_BBOXES,
    expansion_radius_steps=4,
)
WORLDWIDE_TRAINING_BBOXES.extend(WORLDWIDE_EXPANSION_BBOXES)

WORLDWIDE_PILOT_BBOXES = WORLDWIDE_TRAINING_BBOXES[:16]


def _has_usable_name(osm_tags: dict) -> bool:
    name = osm_tags.get("name")

    return isinstance(name, str) and name.strip() != ""


def _has_environmental_tag(osm_tags: dict) -> bool:
    return any(osm_tags.get(key) == value for key, value in ENVIRONMENTAL_TAGS)


def filter_named_environmental_polygons(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if "osm_tags" not in gdf.columns:
        return gdf.head(0).copy()

    keep_mask = gdf["osm_tags"].apply(
        lambda osm_tags: isinstance(osm_tags, dict)
        and _has_usable_name(osm_tags)
        and _has_environmental_tag(osm_tags)
    )

    return gdf[keep_mask].copy().reset_index(drop=True)


def add_bbox_metadata(
    gdf: gpd.GeoDataFrame,
    bbox_config: dict,
) -> gpd.GeoDataFrame:
    gdf = gdf.copy()

    for column in [
        "bbox_id",
        "bbox_label",
        "country",
        "world_region",
        "local_language",
    ]:
        gdf[column] = bbox_config[column]

    return gdf


def add_area_size_bin(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    gdf = gdf.copy()
    gdf["area_size_bin"] = pd.cut(
        gdf["area_km2"],
        bins=[0, 0.1, 1, 10, float("inf")],
        labels=["tiny", "small", "medium", "large"],
        right=False,
        include_lowest=True,
    ).astype(str)

    return gdf


def _balanced_downsample(
    gdf: gpd.GeoDataFrame,
    sample_size: int,
    group_columns: list[str],
    random_state: int | None,
) -> gpd.GeoDataFrame:
    if not group_columns:
        return gdf.sample(n=sample_size, random_state=random_state)

    groups = list(gdf.groupby(group_columns, sort=True, dropna=False))
    base_quota = sample_size // len(groups)
    remainder = sample_size % len(groups)

    sampled_parts = []
    sampled_indices = []
    for group_index, (_, group) in enumerate(groups):
        quota = base_quota + int(group_index < remainder)
        if quota == 0:
            continue

        seed = None if random_state is None else random_state + group_index
        n = min(quota, len(group))
        if "_spatial_priority" in group.columns:
            part = group.sort_values("_spatial_priority").head(n)
        else:
            part = group.sample(n=n, random_state=seed)
        sampled_parts.append(part)
        sampled_indices.extend(part.index.to_list())

    sample = pd.concat(sampled_parts) if sampled_parts else gdf.head(0)

    if len(sample) < sample_size:
        remaining = gdf.drop(index=sampled_indices)
        fill_n = min(sample_size - len(sample), len(remaining))
        if fill_n > 0:
            fill = remaining.sample(n=fill_n, random_state=random_state)
            sample = pd.concat([sample, fill])

    return gpd.GeoDataFrame(
        sample,
        geometry=gdf.geometry.name,
        crs=gdf.crs,
    )


def _cap_group_size(
    gdf: gpd.GeoDataFrame,
    group_column: str,
    max_per_group: int | None,
    random_state: int | None,
) -> gpd.GeoDataFrame:
    if max_per_group is None or group_column not in gdf.columns:
        return gdf

    if max_per_group <= 0:
        return gdf.head(0).copy()

    capped_parts = []
    for group_index, (_, group) in enumerate(gdf.groupby(group_column, sort=True)):
        seed = None if random_state is None else random_state + group_index
        n = min(max_per_group, len(group))
        capped_parts.append(group.sample(n=n, random_state=seed))

    capped = pd.concat(capped_parts) if capped_parts else gdf.head(0)

    return gpd.GeoDataFrame(
        capped,
        geometry=gdf.geometry.name,
        crs=gdf.crs,
    )


def _group_columns_for_balancing(gdf: gpd.GeoDataFrame) -> list[str]:
    return [
        column
        for column in ["world_region", "area_size_bin"]
        if column in gdf.columns
    ]


def _group_key(row: pd.Series, group_columns: list[str]) -> tuple:
    if not group_columns:
        return ("__all__",)

    return tuple(row[column] for column in group_columns)


def _compute_group_targets(
    gdf: gpd.GeoDataFrame,
    sample_size: int,
    group_columns: list[str],
) -> dict[tuple, int]:
    if not group_columns:
        return {("__all__",): min(sample_size, len(gdf))}

    groups = list(gdf.groupby(group_columns, sort=True, dropna=False))
    group_sizes = {
        group_key if isinstance(group_key, tuple) else (group_key,): len(group)
        for group_key, group in groups
    }
    targets = dict.fromkeys(group_sizes, 0)

    while sum(targets.values()) < min(sample_size, len(gdf)):
        grew = False
        for group_key in sorted(group_sizes):
            if sum(targets.values()) >= min(sample_size, len(gdf)):
                break

            if targets[group_key] >= group_sizes[group_key]:
                continue

            targets[group_key] += 1
            grew = True

        if not grew:
            break

    return targets


def _is_far_enough_from_points(
    lon: float,
    lat: float,
    points: list[tuple[float, float]],
    min_distance_km: float,
) -> bool:
    return all(
        _geodesic_distance_km(lon, lat, selected_lon, selected_lat)
        >= min_distance_km
        for selected_lon, selected_lat in points
    )


def _distance_cell_size_degrees(min_distance_km: float) -> float:
    if min_distance_km <= 0:
        return 1.0

    return max(min_distance_km / 111, 0.25)


def _distance_cell_key(
    lon: float,
    lat: float,
    cell_size_degrees: float,
) -> tuple[int, int]:
    return (
        math.floor(lat / cell_size_degrees),
        math.floor(lon / cell_size_degrees),
    )


def _nearby_distance_cells(
    lon: float,
    lat: float,
    cell_size_degrees: float,
    min_distance_km: float,
) -> list[tuple[int, int]]:
    lat_cell, lon_cell = _distance_cell_key(lon, lat, cell_size_degrees)
    lat_radius_degrees = min_distance_km / 110.574
    lon_scale = max(111.320 * math.cos(math.radians(lat)), 1)
    lon_radius_degrees = min_distance_km / lon_scale
    lat_steps = math.ceil(lat_radius_degrees / cell_size_degrees)
    lon_steps = math.ceil(lon_radius_degrees / cell_size_degrees)

    return [
        (nearby_lat_cell, nearby_lon_cell)
        for nearby_lat_cell in range(lat_cell - lat_steps, lat_cell + lat_steps + 1)
        for nearby_lon_cell in range(lon_cell - lon_steps, lon_cell + lon_steps + 1)
    ]


def _add_point_to_distance_grid(
    grid: dict[tuple[int, int], list[tuple[float, float]]],
    lon: float,
    lat: float,
    cell_size_degrees: float,
) -> None:
    grid.setdefault(
        _distance_cell_key(lon, lat, cell_size_degrees),
        [],
    ).append((lon, lat))


def _is_far_enough_from_distance_grid(
    lon: float,
    lat: float,
    grid: dict[tuple[int, int], list[tuple[float, float]]],
    cell_size_degrees: float,
    min_distance_km: float,
) -> bool:
    for cell_key in _nearby_distance_cells(
        lon,
        lat,
        cell_size_degrees,
        min_distance_km,
    ):
        if not _is_far_enough_from_points(
            lon,
            lat,
            grid.get(cell_key, []),
            min_distance_km,
        ):
            return False

    return True


def _select_balanced_sparse_rows(
    gdf: gpd.GeoDataFrame,
    sample_size: int,
    max_per_bbox: int | None,
    max_per_country: int | None,
    min_centroid_distance_km: float,
    min_global_centroid_distance_km: float,
    random_state: int | None,
) -> gpd.GeoDataFrame:
    gdf = _add_representative_coordinates(gdf)
    group_columns = _group_columns_for_balancing(gdf)
    group_targets = _compute_group_targets(gdf, sample_size, group_columns)

    grouped_rows = {}
    for group_index, (group_key, group) in enumerate(
        gdf.groupby(group_columns, sort=True, dropna=False)
        if group_columns
        else [(("__all__",), gdf)]
    ):
        normalized_key = group_key if isinstance(group_key, tuple) else (group_key,)
        seed = None if random_state is None else random_state + group_index
        grouped_rows[normalized_key] = list(group.sample(frac=1, random_state=seed).iterrows())

    grouped_positions = dict.fromkeys(grouped_rows, 0)
    selected_rows = []
    selected_indices = set()
    selected_group_counts = dict.fromkeys(group_targets, 0)
    selected_bbox_counts = {}
    selected_country_counts = {}
    selected_bbox_grids = {}
    selected_global_grid = {}
    local_cell_size_degrees = _distance_cell_size_degrees(min_centroid_distance_km)
    global_cell_size_degrees = _distance_cell_size_degrees(
        min_global_centroid_distance_km
    )

    def can_select(
        index,
        row,
        enforce_local_distance: bool,
        enforce_global_distance: bool,
    ) -> bool:
        if index in selected_indices:
            return False

        bbox_id = row.get("bbox_id")
        if (
            max_per_bbox is not None
            and bbox_id is not None
            and selected_bbox_counts.get(bbox_id, 0) >= max_per_bbox
        ):
            return False

        country = row.get("country")
        if (
            max_per_country is not None
            and country is not None
            and selected_country_counts.get(country, 0) >= max_per_country
        ):
            return False

        lon = row["centroid_lon"]
        lat = row["centroid_lat"]
        if (
            enforce_local_distance
            and min_centroid_distance_km > 0
            and bbox_id is not None
            and not _is_far_enough_from_distance_grid(
                lon,
                lat,
                selected_bbox_grids.get(bbox_id, {}),
                local_cell_size_degrees,
                min_centroid_distance_km,
            )
        ):
            return False

        if (
            enforce_global_distance
            and min_global_centroid_distance_km > 0
            and not _is_far_enough_from_distance_grid(
                lon,
                lat,
                selected_global_grid,
                global_cell_size_degrees,
                min_global_centroid_distance_km,
            )
        ):
            return False

        return True

    def select(index, row) -> None:
        selected_rows.append(row)
        selected_indices.add(index)
        group_key = _group_key(row, group_columns)
        selected_group_counts[group_key] = selected_group_counts.get(group_key, 0) + 1

        bbox_id = row.get("bbox_id")
        if bbox_id is not None:
            selected_bbox_counts[bbox_id] = selected_bbox_counts.get(bbox_id, 0) + 1
            selected_bbox_grids.setdefault(bbox_id, {})
            _add_point_to_distance_grid(
                selected_bbox_grids[bbox_id],
                row["centroid_lon"],
                row["centroid_lat"],
                local_cell_size_degrees,
            )

        country = row.get("country")
        if country is not None:
            selected_country_counts[country] = selected_country_counts.get(country, 0) + 1

        _add_point_to_distance_grid(
            selected_global_grid,
            row["centroid_lon"],
            row["centroid_lat"],
            global_cell_size_degrees,
        )

    def pick_next_for_group(
        group_key: tuple,
        enforce_local_distance: bool,
        enforce_global_distance: bool,
    ) -> bool:
        rows = grouped_rows.get(group_key, [])
        position = grouped_positions.get(group_key, 0)
        while position < len(rows):
            index, row = rows[position]
            grouped_positions[group_key] = position + 1
            position += 1
            if can_select(index, row, enforce_local_distance, enforce_global_distance):
                select(index, row)
                return True

        return False

    while len(selected_rows) < min(sample_size, len(gdf)):
        eligible_group_keys = [
            group_key
            for group_key, target in group_targets.items()
            if selected_group_counts.get(group_key, 0) < target
        ]
        if not eligible_group_keys:
            break

        eligible_group_keys.sort(
            key=lambda group_key: (
                selected_group_counts.get(group_key, 0) / max(group_targets[group_key], 1),
                group_key,
            )
        )

        made_progress = False
        for group_key in eligible_group_keys:
            if pick_next_for_group(
                group_key,
                enforce_local_distance=True,
                enforce_global_distance=True,
            ):
                made_progress = True
                break

        if not made_progress:
            break

    if len(selected_rows) < min(sample_size, len(gdf)):
        shuffled = gdf.sample(frac=1, random_state=random_state)
        for index, row in shuffled.iterrows():
            if len(selected_rows) >= min(sample_size, len(gdf)):
                break

            if can_select(
                index,
                row,
                enforce_local_distance=False,
                enforce_global_distance=True,
            ):
                select(index, row)

    selected = (
        gpd.GeoDataFrame(selected_rows, geometry=gdf.geometry.name, crs=gdf.crs)
        if selected_rows
        else gdf.head(0).copy()
    )

    return selected


def _add_representative_coordinates(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if {"centroid_lon", "centroid_lat"}.issubset(gdf.columns):
        return gdf

    gdf = gdf.copy()
    representative_points = gdf.geometry.apply(lambda geometry: geometry.representative_point())
    gdf["centroid_lon"] = representative_points.apply(lambda point: point.x)
    gdf["centroid_lat"] = representative_points.apply(lambda point: point.y)

    return gdf


def _geodesic_distance_km(
    lon_a: float,
    lat_a: float,
    lon_b: float,
    lat_b: float,
) -> float:
    _, _, distance_m = WGS84_GEOD.inv(lon_a, lat_a, lon_b, lat_b)

    return distance_m / 1_000


def _spatially_thin(
    gdf: gpd.GeoDataFrame,
    target_size: int,
    min_centroid_distance_km: float,
    random_state: int | None,
    fill_shortfall: bool = True,
) -> gpd.GeoDataFrame:
    if min_centroid_distance_km <= 0:
        n = min(target_size, len(gdf))
        sample = gdf.sample(n=n, random_state=random_state)
        sample["_spatial_priority"] = 0
        return sample

    gdf = _add_representative_coordinates(gdf)
    shuffled = gdf.sample(frac=1, random_state=random_state)

    selected_rows = []
    selected_indices = []
    selected_points = []
    for index, row in shuffled.iterrows():
        lon = row["centroid_lon"]
        lat = row["centroid_lat"]
        is_far_enough = all(
            _geodesic_distance_km(lon, lat, selected_lon, selected_lat)
            >= min_centroid_distance_km
            for selected_lon, selected_lat in selected_points
        )

        if not is_far_enough:
            continue

        selected_rows.append(row)
        selected_indices.append(index)
        selected_points.append((lon, lat))

        if len(selected_rows) >= target_size:
            break

    selected = (
        gpd.GeoDataFrame(selected_rows, geometry=gdf.geometry.name, crs=gdf.crs)
        if selected_rows
        else gdf.head(0)
    )
    selected["_spatial_priority"] = 0

    if fill_shortfall and len(selected) < min(target_size, len(gdf)):
        remaining = shuffled.drop(index=selected_indices)
        fill_n = min(target_size - len(selected), len(remaining))
        if fill_n > 0:
            fill = remaining.sample(n=fill_n, random_state=random_state)
            fill["_spatial_priority"] = 1
            selected = pd.concat([selected, fill])

    return gpd.GeoDataFrame(
        selected,
        geometry=gdf.geometry.name,
        crs=gdf.crs,
    )


def sample_worldwide_polygons(
    gdf: gpd.GeoDataFrame,
    sample_size: int = 100,
    max_per_bbox: int = 8,
    max_per_country: int | None = None,
    min_centroid_distance_km: float = 0,
    min_global_centroid_distance_km: float = 0,
    random_state: int = 42,
) -> gpd.GeoDataFrame:
    if gdf.empty:
        return gdf.copy()

    sample = _select_balanced_sparse_rows(
        gdf,
        sample_size=sample_size,
        max_per_bbox=max_per_bbox,
        max_per_country=max_per_country,
        min_centroid_distance_km=min_centroid_distance_km,
        min_global_centroid_distance_km=min_global_centroid_distance_km,
        random_state=random_state,
    )

    if "_spatial_priority" in sample.columns:
        sample = sample.drop(columns=["_spatial_priority"])

    return sample.reset_index(drop=True)
