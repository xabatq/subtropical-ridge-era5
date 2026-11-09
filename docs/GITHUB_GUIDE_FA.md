# راهنمای انتشار روی GitHub و Zenodo (فارسی)

## ۱. نام و سربرگ پیشنهادی
- **نام مخزن:** `subtropical-ridge-era5`
- **توضیح کوتاه (About):** Code for ocean subtropical-ridge latitude trends in ERA5 (1966–2025) with CERES and CLARA-A3 checks
- **Topics:** `climate-dynamics` `era5` `hadley-cell` `subtropical-ridge` `reanalysis` `ceres` `clara-a3` `python` `atmospheric-science` `trend-analysis`
- **Website:** پس از دریافت DOI، لینک Zenodo را بگذارید.

## ۲. فایل‌های لازم (همه در همین بسته هست)
README.md، LICENSE، CITATION.cff، .zenodo.json، requirements.txt، environment.yml، .gitignore، scripts/ (۱۵ اسکریپت + run_all.py)، tools/، docs/ (DATA.md، PIPELINE.md)، data/README.md، results/ (جدول‌های CSV نهایی).
**قبل از آپلود:** پس از اجرای کامل روی داده‌های خودتان، CSVهای فهرست‌شده در `results/README.md` را از پوشهٔ `outputs` به `results/` کپی کنید. فایل‌های بزرگ (`.nc`، `.npz`) را آپلود نکنید (در .gitignore هستند).

## ۳. آپلود (روش ساده، بدون git)
1. در github.com وارد شوید ← New repository ← نام بالا ← Public ← بدون README/LICENSE (چون خودمان داریم) ← Create.
2. روی «uploading an existing file» بزنید، محتویات پوشهٔ بسته (نه خود پوشه) را بکشید و رها کنید ← Commit changes.
(اگر پوشه‌ها آپلود نشدند، از GitHub Desktop یا دستورات زیر استفاده کنید.)

```bash
git init && git add . && git commit -m "Initial release"
git branch -M main
git remote add origin https://github.com/USERNAME/subtropical-ridge-era5.git
git push -u origin main
```

## ۴. جایگزینی نام کاربری
در `README.md`، `CITATION.cff`، و متن مقاله عبارت `YOUR-USERNAME` را با نام کاربری GitHub خود عوض کنید.

## ۵. DOI از Zenodo
1. در zenodo.org با حساب GitHub وارد شوید ← Settings ← GitHub ← کلید مخزن را روشن کنید.
2. در GitHub: Releases ← Draft a new release ← تگ `v1.0.0` ← Publish.
3. Zenodo خودکار آرشیو می‌کند و DOI می‌دهد (DOI «concept» برای همهٔ نسخه‌ها، ثابت است؛ همان را در مقاله بگذارید).
4. بعد از دریافت DOI: در `CITATION.cff` خط doi را فعال کنید، در `README.md` مقدار `10.5281/zenodo.XXXXXXX` را عوض کنید، و در مقاله `\todo{repository DOI}` را جایگزین کنید. (یک release جدید لازم نیست؛ فقط README را به‌روز کنید، یا برای دقیق‌تر بودن v1.0.1 بزنید.)
نکته: release را **پس از** پایان تغییرات مقاله (پس از داوری) و پس از کپی جدول‌ها در `results/` بزنید.

## ۶. جملهٔ Code availability پیشنهادی
The analysis scripts (Python), a pipeline runner and all derived tables (CSV) are available at https://github.com/USERNAME/subtropical-ridge-era5 and archived on Zenodo (https://doi.org/10.5281/zenodo.XXXXXXX).

## ۷. نکته‌ها
- ضمیمه (Supplementary) را می‌توان حذف کرد، به شرط آنکه CSVهای نهایی داخل `results/` و آرشیو Zenodo باشند (مقاله اکنون به همین اشاره می‌کند). برخی مجلات ضمیمهٔ جدا را ترجیح می‌دهند؛ در صورت درخواست داور/سردبیر، همان CSVها را آپلود کنید.
- پیش از انتشار، `python scripts/run_all.py --list` را اجرا کنید تا ترتیب مراحل را ببینید.
- اسکریپت `00_make_static_fields.py` جدید است و فقط روی داده‌های مصنوعی آزمایش شده؛ اگر اسکریپت قبلی خودتان کار می‌کند، همان را جایگزین کنید (خروجی: `outputs/static_fields.nc` با متغیرهای lsm و height_m).
