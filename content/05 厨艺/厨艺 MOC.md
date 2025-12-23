可能，做饭真的是我一种天生的爱好。

```dataview
TABLE
    file.name AS "📅 日期",
    ☁️天气☁️ AS "☁️ 天气",
    🌡️温度🌡️ AS "🌡️ 温度",
    湿度 AS "💧 湿度"
FROM "05 厨艺"
WHERE file.name != "Diary MOC"
SORT file.name DESC
```
