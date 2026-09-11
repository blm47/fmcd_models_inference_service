# Ресурсы автономного Swagger UI

Swagger UI Dist 5.17.14, лицензия Apache-2.0 (см. LICENSE).
FastAPI также использует Swagger UI 5 для штатной страницы `/docs`.

Исходные файлы:

- https://cdn.jsdelivr.net/npm/swagger-ui-dist@5.17.14/swagger-ui-bundle.js
- https://cdn.jsdelivr.net/npm/swagger-ui-dist@5.17.14/swagger-ui.css
- https://cdn.jsdelivr.net/npm/swagger-ui-dist@5.17.14/LICENSE

`scripts/export_openapi.py` встраивает JS и CSS в `docs/swagger.html`.
Повторный экспорт не требует сети. При обновлении библиотеки заменяйте оба
ресурса и лицензию из одной версии, обновляйте этот файл и повторяйте экспорт.
