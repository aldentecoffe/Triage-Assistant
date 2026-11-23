"""Доменная модель учебного набора инцидентов.

Модуль содержит только данные: каталог сервисов, категории корневых причин,
сценарии отказов, каскады и правило критичности по формулировке последствия.
Генератор, очистка и split берут значения отсюда, документы в docs/lab4
ссылаются на этот же файл.
"""

GENERATOR_VERSION = "1.1"
DATASET_VERSION = "1.1"

SERVICES = {
    "transport": {
        "ru": "Транспортная сеть",
        "en": "Transport network",
        "owner": "Сеть",
    },
    "platform": {
        "ru": "Платформа Kubernetes",
        "en": "Kubernetes platform",
        "owner": "Платформа",
    },
    "database": {
        "ru": "СУБД",
        "en": "Database",
        "owner": "Платформа данных",
    },
    "ims": {
        "ru": "IMS/VoLTE",
        "en": "IMS/VoLTE",
        "owner": "Голосовые сервисы",
    },
    "billing": {
        "ru": "Биллинг",
        "en": "Billing",
        "owner": "Биллинг",
    },
    "bss": {
        "ru": "BSS/OSS",
        "en": "BSS/OSS",
        "owner": "Бизнес-системы",
    },
    "external": {
        "ru": "Внешние интеграции",
        "en": "External integrations",
        "owner": "Интеграции",
    },
}

ROOT_CAUSES = {
    "change": "Изменение или релиз",
    "capacity": "Нехватка ёмкости",
    "hardware": "Отказ оборудования",
    "software": "Дефект ПО",
    "network": "Сеть и маршрутизация",
    "external_dep": "Внешний поставщик",
}

SEVERITIES = ("P1", "P2", "P3", "P4")

# Критичность выводится из формулировки последствия, а не разыгрывается.
# Правило версии 1.1, полное описание в docs/lab6/annotation_guideline.md:
#   P1 — ключевая клиентская функция не выполняется (голос и регистрация в сети,
#        платежи, доступность магистрали), резервного пути нет;
#   P2 — функция не выполняется для части объектов, недоступна отдельная функция
#        или интеграция, клиент видит ошибку или неверный результат;
#   P3 — ограниченное влияние: задержка, очередь, часть ассортимента, есть
#        обходной путь (резерв, повтор, ручное оформление, отложенная обработка);
#   P4 — абоненты и клиентские операции не затронуты: мониторинг, отчётность,
#        справочники, автоматическое восстановление.
# Значение — критичность каждого последствия сценария: сначала два русских
# варианта в порядке impacts_ru, затем английский.
IMPACT_SEVERITY = {
    "SC-TRN-01": {"ru": ("P3", "P3"), "en": "P3"},
    "SC-TRN-02": {"ru": ("P2", "P2"), "en": "P2"},
    "SC-TRN-03": {"ru": ("P3", "P3"), "en": "P3"},
    "SC-TRN-04": {"ru": ("P3", "P2"), "en": "P2"},
    "SC-PLT-01": {"ru": ("P3", "P4"), "en": "P3"},
    "SC-PLT-02": {"ru": ("P3", "P3"), "en": "P3"},
    "SC-PLT-03": {"ru": ("P4", "P3"), "en": "P4"},
    "SC-PLT-04": {"ru": ("P2", "P3"), "en": "P2"},
    "SC-DB-01": {"ru": ("P2", "P2"), "en": "P2"},
    "SC-DB-02": {"ru": ("P4", "P3"), "en": "P4"},
    "SC-DB-03": {"ru": ("P2", "P3"), "en": "P2"},
    "SC-DB-04": {"ru": ("P2", "P3"), "en": "P2"},
    "SC-IMS-01": {"ru": ("P1", "P3"), "en": "P1"},
    "SC-IMS-02": {"ru": ("P3", "P3"), "en": "P3"},
    "SC-IMS-03": {"ru": ("P1", "P3"), "en": "P1"},
    "SC-IMS-04": {"ru": ("P2", "P2"), "en": "P2"},
    "SC-BIL-01": {"ru": ("P2", "P4"), "en": "P2"},
    "SC-BIL-02": {"ru": ("P2", "P3"), "en": "P2"},
    "SC-BIL-03": {"ru": ("P1", "P1"), "en": "P1"},
    "SC-BIL-04": {"ru": ("P4", "P3"), "en": "P4"},
    "SC-BSS-01": {"ru": ("P2", "P3"), "en": "P2"},
    "SC-BSS-02": {"ru": ("P2", "P3"), "en": "P2"},
    "SC-BSS-03": {"ru": ("P3", "P4"), "en": "P3"},
    "SC-BSS-04": {"ru": ("P2", "P3"), "en": "P2"},
    "SC-EXT-01": {"ru": ("P2", "P2"), "en": "P2"},
    "SC-EXT-02": {"ru": ("P3", "P3"), "en": "P3"},
    "SC-EXT-03": {"ru": ("P2", "P2"), "en": "P2"},
    "SC-EXT-04": {"ru": ("P3", "P3"), "en": "P3"},
}

REGIONS = (
    ("Москва", "msk"),
    ("Санкт-Петербург", "spb"),
    ("Новосибирск", "nsk"),
    ("Екатеринбург", "ekb"),
    ("Казань", "kzn"),
    ("Краснодар", "krd"),
    ("Хабаровск", "khv"),
)

MONITORING_SOURCES = (
    "Zabbix",
    "Prometheus",
    "Grafana",
    "Netcool",
    "SCOM",
    "Nagios",
    "Dynatrace",
)

RU_SHARE = 0.8

# Метрика описана кортежем: (имя, порог, единица, направление).
# Направление "high" означает, что плохо выше порога, "low" — ниже.
SCENARIOS = [
    {
        "id": "SC-TRN-01",
        "service": "transport",
        "root_cause": "network",
        "split": "train",
        "symptoms_ru": (
            "падает BGP-сессия с пирингом",
            "BGP-сессия с апстримом уходит в Idle",
        ),
        "symptom_en": "BGP session with the upstream peer keeps flapping",
        "impacts_ru": (
            "около 400 префиксов ушло в резерв, задержка на магистральном направлении выросла",
            "часть клиентских префиксов анонсируется через резервный аплинк",
        ),
        "impact_en": "some prefixes are withdrawn and traffic shifts to the backup uplink",
        "metrics": (
            ("BGP_Peers_Established", 2, "шт", "low"),
            ("route_withdrawals", 50, "шт/мин", "high"),
        ),
    },
    {
        "id": "SC-TRN-02",
        "service": "transport",
        "root_cause": "capacity",
        "split": "train",
        "symptoms_ru": (
            "таблица маршрутизации упирается в лимит FIB",
            "коммутатор теряет FIB-записи под нагрузкой",
        ),
        "symptom_en": "FIB table has reached its limit on the aggregation switch",
        "impacts_ru": (
            "новые префиксы не устанавливаются, часть трафика уходит в чёрную дыру",
            "маршруты до части площадок пропадают на минуты",
        ),
        "impact_en": "new prefixes are rejected and part of the traffic is black-holed",
        "metrics": (
            ("fib_usage_pct", 92, "%", "high"),
            ("prefix_count", 2400000, "шт", "high"),
        ),
    },
    {
        "id": "SC-TRN-03",
        "service": "transport",
        "root_cause": "hardware",
        "split": "validation",
        "symptoms_ru": (
            "линейная карта на агрегации переходит в ошибку",
            "порт 100GE на агрегационном узле флапает",
        ),
        "symptom_en": "line card on the aggregation router reports a hardware fault",
        "impacts_ru": (
            "кольцо агрегации в деградации, потеряна половина ёмкости",
            "часть площадок переключилась на резервный путь",
        ),
        "impact_en": "aggregation ring is degraded and half of the capacity is lost",
        "metrics": (
            ("if_errors_crc", 1000, "шт/мин", "high"),
            ("link_uptime", 60, "с", "low"),
        ),
    },
    {
        "id": "SC-TRN-04",
        "service": "transport",
        "root_cause": "network",
        "split": "test",
        "symptoms_ru": (
            "после правки ACL появилась петля маршрутизации",
            "TTL истекает на транзитном узле, трафик зациклился",
        ),
        "symptom_en": "routing loop after an ACL change, TTL expires on the transit node",
        "impacts_ru": (
            "запросы к BSS идут с задержкой 4-6 с",
            "часть сессий до биллинга обрывается по таймауту",
        ),
        "impact_en": "requests to BSS are delayed by 4-6 s and some sessions time out",
        "metrics": (
            ("traceroute_hops", 24, "хопов", "high"),
            ("icmp_ttl_exceeded", 800, "шт/с", "high"),
        ),
    },
    {
        "id": "SC-PLT-01",
        "service": "platform",
        "root_cause": "hardware",
        "split": "test",
        "symptoms_ru": (
            "гипервизор переходит в состояние degraded",
            "кластер теряет один хост, виртуальные машины мигрируют",
        ),
        "symptom_en": "hypervisor goes into degraded state and VMs migrate",
        "impacts_ru": (
            "около 30 виртуальных машин мигрировали, часть подов перезапустилась",
            "нагрузка перераспределилась на оставшиеся хосты",
        ),
        "impact_en": "about 30 VMs migrated and some pods restarted",
        "metrics": (
            ("esxi_hosts_online", 12, "шт", "low"),
            ("vm_migration_count", 5, "шт/мин", "high"),
        ),
    },
    {
        "id": "SC-PLT-02",
        "service": "platform",
        "root_cause": "capacity",
        "split": "train",
        "symptoms_ru": (
            "в кластере закончились свободные адреса в пуле",
            "kubelet не может выдать адрес новым подам",
        ),
        "symptom_en": "cluster IP pool is exhausted and new pods stay pending",
        "impacts_ru": (
            "новые деплои не раскатываются, автомасштабирование не работает",
            "часть подов висит в Pending, релиз остановлен",
        ),
        "impact_en": "new deployments stall and autoscaling stops working",
        "metrics": (
            ("ip_pool_free", 3, "%", "low"),
            ("pod_pending", 40, "шт", "high"),
        ),
    },
    {
        "id": "SC-PLT-03",
        "service": "platform",
        "root_cause": "software",
        "split": "validation",
        "symptoms_ru": (
            "агент мониторинга падает на части узлов",
            "прокси-компонент сбрасывает соединения под нагрузкой",
        ),
        "symptom_en": "the monitoring agent crashes on part of the nodes",
        "impacts_ru": (
            "метрики части узлов пропали, алерты не срабатывают",
            "часть соединений сбрасывается, растёт число ретраев",
        ),
        "impact_en": "metrics from some nodes are missing and alerts do not fire",
        "metrics": (
            ("agent_restarts", 12, "шт/час", "high"),
            ("conntrack_drops", 300, "шт/мин", "high"),
        ),
    },
    {
        "id": "SC-PLT-04",
        "service": "platform",
        "root_cause": "change",
        "split": "train",
        "symptoms_ru": (
            "после релиза конфигурации поды уходят в OOMKilled",
            "лимиты памяти занижены, контейнеры падают",
        ),
        "symptom_en": "pods are killed with OOMKilled after the configuration release",
        "impacts_ru": (
            "сервис заказов частично недоступен",
            "часть реплик перезапускается по кругу",
        ),
        "impact_en": "the order service is partly unavailable",
        "metrics": (
            ("oom_kills", 18, "шт/час", "high"),
            ("memory_usage_pct", 92, "%", "high"),
        ),
    },
    {
        "id": "SC-DB-01",
        "service": "database",
        "root_cause": "capacity",
        "split": "validation",
        "symptoms_ru": (
            "растёт число ожиданий на блокировках",
            "очередь блокировок в СУБД не рассасывается",
        ),
        "symptom_en": "lock waiters in the database keep growing",
        "impacts_ru": (
            "запросы API заказов висят по 30 секунд",
            "часть транзакций отваливается по таймауту",
        ),
        "impact_en": "order API requests hang for up to 30 s",
        "metrics": (
            ("lock_waiters", 60, "сессий", "high"),
            ("avg_lock_wait_ms", 4000, "мс", "high"),
        ),
    },
    {
        "id": "SC-DB-03",
        "service": "database",
        "root_cause": "software",
        "split": "test",
        "symptoms_ru": (
            "СУБД возвращает ошибки исполнения на части запросов",
            "СУБД сбрасывает сессии при выполнении части запросов",
        ),
        "symptom_en": "the database returns execution errors for part of the queries",
        "impacts_ru": (
            "часть запросов к биллингу завершается ошибкой",
            "реплика отстала от мастера и не догоняет",
        ),
        "impact_en": "some billing queries fail and the replica lags behind",
        "metrics": (
            ("db_errors", 25, "шт/мин", "high"),
            ("session_reset", 40, "шт/мин", "high"),
        ),
    },
    {
        "id": "SC-DB-02",
        "service": "database",
        "root_cause": "hardware",
        "split": "train",
        "symptoms_ru": (
            "диск в RAID-массиве переходит в predictive failure",
            "растёт задержка ввода-вывода на массиве",
        ),
        "symptom_en": "RAID disk reports predictive failure and IO latency grows",
        "impacts_ru": (
            "реплика отстала, отчёты не строятся",
            "часть запросов деградировала до секунд",
        ),
        "impact_en": "the replica lags behind and reports cannot be built",
        "metrics": (
            ("io_latency_ms", 120, "мс", "high"),
            ("raid_degraded_disks", 1, "шт", "high"),
        ),
    },
    {
        "id": "SC-DB-04",
        "service": "database",
        "root_cause": "change",
        "split": "train",
        "symptoms_ru": (
            "миграция схемы не завершилась, таблица заблокирована",
            "после миграции схемы выросли ошибки запросов",
        ),
        "symptom_en": "schema migration did not finish and the table stays locked",
        "impacts_ru": (
            "оформление заказов остановлено",
            "часть фоновых задач встала в очередь",
        ),
        "impact_en": "order processing is stopped",
        "metrics": (
            ("migration_duration_min", 45, "мин", "high"),
            ("sql_errors", 90, "шт/мин", "high"),
        ),
    },
    {
        "id": "SC-IMS-01",
        "service": "ims",
        "root_cause": "capacity",
        "split": "train",
        "symptoms_ru": (
            "растёт доля отказов IMS REGISTER",
            "CSCF не успевает обрабатывать регистрации",
        ),
        "symptom_en": "IMS REGISTER failure rate is growing",
        "impacts_ru": (
            "абоненты в зоне обслуживания не могут зарегистрироваться в сети",
            "часть абонентов уходит в 3G при попытке звонка",
        ),
        "impact_en": "subscribers in the serving area cannot register in the network",
        "metrics": (
            ("register_fail_pct", 8, "%", "high"),
            ("cscf_cpu_pct", 96, "%", "high"),
        ),
    },
    {
        "id": "SC-IMS-02",
        "service": "ims",
        "root_cause": "hardware",
        "split": "train",
        "symptoms_ru": (
            "нода HSS выпала из кластера",
            "кластер HSS потерял одну из реплик",
        ),
        "symptom_en": "HSS node dropped out of the cluster",
        "impacts_ru": (
            "часть абонентов теряет VoLTE при переключении",
            "рост таймаутов Diameter на регистрации",
        ),
        "impact_en": "some subscribers lose VoLTE during handover",
        "metrics": (
            ("hss_nodes_online", 2, "шт", "low"),
            ("diameter_timeouts", 120, "шт/мин", "high"),
        ),
    },
    {
        "id": "SC-IMS-03",
        "service": "ims",
        "root_cause": "software",
        "split": "validation",
        "symptoms_ru": (
            "граничный контроллер обрывает вызовы на части попыток",
            "граничный контроллер сбрасывает вызовы при переключении",
        ),
        "symptom_en": "the edge controller drops calls on part of the attempts",
        "impacts_ru": (
            "обрывы звонков у абонентов зоны обслуживания",
            "часть вызовов не переключается на VoWiFi",
        ),
        "impact_en": "calls drop for subscribers in the serving area",
        "metrics": (
            ("call_drop_pct", 6, "%", "high"),
            ("sip_errors", 200, "шт/мин", "high"),
        ),
    },
    {
        "id": "SC-IMS-04",
        "service": "ims",
        "root_cause": "capacity",
        "split": "test",
        "symptoms_ru": (
            "после массового перезапуска терминалов всплеск регистраций",
            "пик регистраций превышает ёмкость CSCF",
        ),
        "symptom_en": "registration burst after a mass terminal restart exceeds capacity",
        "impacts_ru": (
            "регистрация абонентов занимает до 40 секунд",
            "часть попыток звонка завершается отказом",
        ),
        "impact_en": "subscriber registration takes up to 40 s",
        "metrics": (
            ("register_rps", 9000, "шт/с", "high"),
            ("cscf_queue_depth", 1500, "шт", "high"),
        ),
    },
    {
        "id": "SC-BIL-01",
        "service": "billing",
        "root_cause": "capacity",
        "split": "train",
        "symptoms_ru": (
            "очередь записей о звонках в биллинге растёт",
            "обработчик записей не успевает за потоком",
        ),
        "symptom_en": "CDR queue in billing keeps growing",
        "impacts_ru": (
            "начисления задерживаются, абоненты видят нулевой баланс",
            "сверка за час не закрылась",
        ),
        "impact_en": "charges are delayed and subscribers see a zero balance",
        "metrics": (
            ("cdr_queue_depth", 250000, "шт", "high"),
            ("cdr_lag_min", 25, "мин", "high"),
        ),
    },
    {
        "id": "SC-BIL-02",
        "service": "billing",
        "root_cause": "change",
        "split": "validation",
        "symptoms_ru": (
            "после релиза правил тарификации пошли двойные списания",
            "новые правила тарификации считают пакеты неверно",
        ),
        "symptom_en": "double charges appear after the rating rules release",
        "impacts_ru": (
            "абоненты получают некорректные счета",
            "в поддержку пошли обращения по списаниям",
        ),
        "impact_en": "subscribers receive incorrect invoices",
        "metrics": (
            ("double_charge_events", 300, "шт/час", "high"),
            ("rating_errors", 90, "шт/мин", "high"),
        ),
    },
    {
        "id": "SC-BIL-03",
        "service": "billing",
        "root_cause": "external_dep",
        "split": "test",
        "symptoms_ru": (
            "платёжный шлюз отвечает таймаутом",
            "провайдер платежей возвращает 503 на все запросы",
        ),
        "symptom_en": "payment gateway responds with a timeout",
        "impacts_ru": (
            "платежи не проходят, растёт очередь повторных попыток",
            "пополнения баланса не зачисляются",
        ),
        "impact_en": "payments fail and the retry queue grows",
        "metrics": (
            ("payment_gateway_errors", 500, "шт/мин", "high"),
            ("payment_success_pct", 35, "%", "low"),
        ),
    },
    {
        "id": "SC-BIL-04",
        "service": "billing",
        "root_cause": "software",
        "split": "train",
        "symptoms_ru": (
            "расчётный модуль считает абонентскую плату с ошибкой",
            "расчётный модуль выдаёт несходящиеся суммы",
        ),
        "symptom_en": "the billing module miscalculates the monthly fee",
        "impacts_ru": (
            "сверка за сутки не сходится",
            "часть счетов придётся пересчитать",
        ),
        "impact_en": "daily reconciliation does not match",
        "metrics": (
            ("calc_mismatch", 120, "шт/час", "high"),
            ("rounding_errors", 40, "шт/час", "high"),
        ),
    },
    {
        "id": "SC-BSS-01",
        "service": "bss",
        "root_cause": "external_dep",
        "split": "validation",
        "symptoms_ru": (
            "CRM отвечает таймаутом на запросы BSS",
            "интеграция с CRM отдаёт 504",
        ),
        "symptom_en": "CRM responds with a timeout to BSS requests",
        "impacts_ru": (
            "операторы не могут оформить заказ",
            "карточка клиента не открывается в BSS",
        ),
        "impact_en": "operators cannot place orders",
        "metrics": (
            ("crm_timeout_pct", 18, "%", "high"),
            ("crm_response_ms", 9000, "мс", "high"),
        ),
    },
    {
        "id": "SC-BSS-02",
        "service": "bss",
        "root_cause": "capacity",
        "split": "train",
        "symptoms_ru": (
            "API заказов не держит нагрузку",
            "пул потоков API заказов переполнен",
        ),
        "symptom_en": "order API cannot handle the load, the thread pool is full",
        "impacts_ru": (
            "заказы встают в очередь, клиенты видят ошибку",
            "часть заявок пришлось оформить вручную",
        ),
        "impact_en": "orders queue up and clients see an error",
        "metrics": (
            ("order_api_p95_ms", 6000, "мс", "high"),
            ("thread_pool_busy_pct", 99, "%", "high"),
        ),
    },
    {
        "id": "SC-BSS-03",
        "service": "bss",
        "root_cause": "change",
        "split": "test",
        "symptoms_ru": (
            "после правки справочника синхронизация не прошла",
            "обновление справочника услуг разъехалось между системами",
        ),
        "symptom_en": "catalog synchronisation failed after the manual change",
        "impacts_ru": (
            "часть тарифов недоступна в продаже",
            "в BSS и CRM разные версии справочника",
        ),
        "impact_en": "some tariffs are unavailable for sale",
        "metrics": (
            ("sync_failed_records", 4000, "шт", "high"),
            ("catalog_revision_diff", 12, "шт", "high"),
        ),
    },
    {
        "id": "SC-BSS-04",
        "service": "bss",
        "root_cause": "hardware",
        "split": "train",
        "symptoms_ru": (
            "узел очереди перестал отвечать",
            "кластер очереди потерял кворум",
        ),
        "symptom_en": "queue node stopped responding, the cluster lost quorum",
        "impacts_ru": (
            "заказы не передаются в биллинг",
            "часть сообщений пришлось переотправить вручную",
        ),
        "impact_en": "orders are not passed to billing",
        "metrics": (
            ("rabbit_nodes_online", 1, "шт", "low"),
            ("queue_unacked", 90000, "шт", "high"),
        ),
    },
    {
        "id": "SC-EXT-01",
        "service": "external",
        "root_cause": "external_dep",
        "split": "train",
        "symptoms_ru": (
            "SMS-шлюз не принимает сообщения",
            "провайдер SMS отдаёт ошибку авторизации",
        ),
        "symptom_en": "SMS gateway rejects messages, the provider returns an auth error",
        "impacts_ru": (
            "коды подтверждения не доходят до абонентов",
            "регистрация в личном кабинете не завершается",
        ),
        "impact_en": "verification codes do not reach subscribers",
        "metrics": (
            ("sms_delivery_fail_pct", 45, "%", "high"),
            ("sms_gateway_errors", 700, "шт/мин", "high"),
        ),
    },
    {
        "id": "SC-EXT-02",
        "service": "external",
        "root_cause": "capacity",
        "split": "train",
        "symptoms_ru": (
            "поставщик ограничил частоту запросов",
            "API поставщика отвечает 429",
        ),
        "symptom_en": "provider throttles API calls and returns 429",
        "impacts_ru": (
            "часть операций интеграции откладывается",
            "очередь отложенных запросов растёт",
        ),
        "impact_en": "some integration operations are postponed",
        "metrics": (
            ("provider_429_pct", 30, "%", "high"),
            ("provider_rps", 120, "запросов/с", "high"),
        ),
    },
    {
        "id": "SC-EXT-03",
        "service": "external",
        "root_cause": "change",
        "split": "validation",
        "symptoms_ru": (
            "после ротации сертификат на шлюзе просрочен",
            "после ротации сертификата TLS-соединение не устанавливается",
        ),
        "symptom_en": "certificate is expired after rotation, TLS handshake fails",
        "impacts_ru": (
            "интеграция с внешней системой недоступна",
            "часть заявок не уходит поставщику",
        ),
        "impact_en": "the integration with the external system is unavailable",
        "metrics": (
            ("tls_handshake_errors", 150, "шт/мин", "high"),
            ("cert_days_left", -2, "дней", "low"),
        ),
    },
    {
        "id": "SC-EXT-04",
        "service": "external",
        "root_cause": "external_dep",
        "split": "test",
        "symptoms_ru": (
            "адаптер не разбирает новый формат ответа провайдера",
            "поставщик перевёл API на новую версию",
        ),
        "symptom_en": "adapter cannot parse the new provider response format",
        "impacts_ru": (
            "статусы доставки не обновляются",
            "часть операций висит в статусе «в обработке»",
        ),
        "impact_en": "delivery statuses are not updated",
        "metrics": (
            ("parse_errors", 400, "шт/мин", "high"),
            ("adapter_success_pct", 20, "%", "low"),
        ),
    },
]

# Каскады: одно событие даёт инциденты в двух сервисах.
# Пары допустимы только внутри одного split, иначе связанные записи
# разъедутся по train и test — это и есть основная утечка.
CASCADES = (
    ("SC-TRN-01", "SC-BIL-01"),
    ("SC-DB-02", "SC-BIL-01"),
    ("SC-PLT-03", "SC-IMS-03"),
    ("SC-TRN-04", "SC-IMS-04"),
)

SPLIT_SHARES = {"train": 0.6, "validation": 0.2, "test": 0.2}

# Контрольные дефекты добавляются в сырой набор специально,
# чтобы очистка была проверяема.
DEFECTS = {
    "duplicate_text": 3,
    "empty_description": 2,
    "rights_unknown": 3,
}

REQUIRED_COLUMNS = (
    "incident_id",
    "scenario_id",
    "event_id",
    "created_at",
    "source_system",
    "region",
    "affected_ci",
    "title",
    "description",
    "language",
    "severity",
    "service",
    "root_cause",
    "split",
    "rights_status",
    "synthetic",
    "source",
    "generator_version",
    "dataset_version",
    "text_hash",
)
