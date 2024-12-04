# Отчёт

## Часть 1: подготовка данных

Для подготовки данных я использовал tree-sitter, так как модуль ast
можно было бы использовать только для Питона.
Функция, удаляющая комментарии из исходного кода, была по большей части
написана ChatGPT.
Выглядит она сомнительно, кажется, что должен быть более удачный способ
выполнить эту задачу.
Однако это решение работало достаточно хорошо, так что переделывать ничего
я не стал.

Ниже привожу пару примеров данных до и после подготовки:

ДО:
```python
# Код ДО
def analyze(self):
    """Run analysis."""
    precision = 'DP' if self.kernel.datatype == 'double' else 'SP'
    self.calculate_cache_access()

    self.results['max_perf'] = self.conv_perf(self.machine['clock'] * self.cores * \
        self.machine['FLOPs per cycle'][precision]['total'])
```

Код ПОСЛЕ, но с комментариями:
```python
def <extra_id_0>(self):
        """Run analysis."""
        precision = 'DP' if self.kernel.datatype == 'double' else 'SP'
        self.calculate_cache_access()

        self.results['max_perf'] = self.conv_perf(self.machine['clock'] * self.cores * \
            self.machine['FLOPs per cycle'][precision]['total'])
```

и без комментариев:
```python
def <extra_id_0>(self):
        
        precision = 'DP' if self.kernel.datatype == 'double' else 'SP'
        self.calculate_cache_access()

        self.results['max_perf'] = self.conv_perf(self.machine['clock'] * self.cores * \
            self.machine['FLOPs per cycle'][precision]['total'])
```

Функция под индексом 16181 (из исходного датасета после фильтрации
по языку программирования) до:
```python
def _stylesheet_param_dict(paramsDict, kwargsDict):
    """Return a copy of paramsDict, updated with kwargsDict entries, wrapped as
    stylesheet arguments.
    kwargsDict entries with a value of None are ignored.
    """
    # beware of changing mutable default arg
    paramsDict = dict(paramsDict)
    for k, v in kwargsDict.items():
        if v is not None: # None values do not override
            paramsDict[k] = v
    paramsDict = stylesheet_params(**paramsDict)
    return paramsDict
```

и после:
```python
def <extra_id_0>(paramsDict, kwargsDict):
    
    
    paramsDict = dict(paramsDict)
    for k, v in kwargsDict.items():
        if v is not None: 
            paramsDict[k] = v
    paramsDict = stylesheet_params(**paramsDict)
    return paramsDict
```

Было очень удобно, кстати, пользоваться параметром cached_file_name
в методах Dataset, чтобы не генерировать данные по несколько раз подряд.

## Часть 2: генерация имён функций (Python)

В качестве модели я использовал CodeT5+.
На генерацию 1000 имён имён функций с ней уходило по 30-40 минут на моей
машине (по определённым причинам не могу сейчас использовать GPU).
Также замечу, что модель как правило выдавала несколько вариантов имён
функций вместо одного.
Выбирал я самый длинный из них, так как предполагаю, что такое название
будет наиболее детальным.
Ещё, кстати, она может генерировать в качестве названий ключевые слова
или другие синтаксические некорректные варианты.
Происходит это редко, так что я не исправлял это поведение.

Результаты:
* EM (с комментариями): 0.212
* EM (без комментариев): 0.145
* ROUGE1 (с комментариями): 0.509
* ROUGE1 (без): 0.388

Примеры результатов:

Функция с настоящим именем `clear_dag_runs` (#103):
```python
def <extra_id_0>():
    """
    Remove any existing DAG runs for the perf test DAGs.
    """
    session = settings.Session()
    drs = session.query(DagRun).filter(
        DagRun.dag_id.in_(DAG_IDS),
    ).all()
    for dr in drs:
        logging.info('Deleting DagRun :: {}'.format(dr))
        session.delete(dr)
```

Модель предложила имя `delete_dag_runs` (независимо от наличия комментариев).
ROUGE1 = 0,(6).

Функция с настоящим именем `publish_to_target` (#237):
```python
def <extra_id_0>(self, target_arn, message):
        """
        Publish a message to a topic or an endpoint.

        :param target_arn: either a TopicArn or an EndpointArn
        :type target_arn: str
        :param message: the default message you want to send
        :param message: str
        """

        conn = self.get_conn()

        messages = {
            'default': message
        }

        return conn.publish(
            TargetArn=target_arn,
            Message=json.dumps(messages),
            MessageStructure='json'
        )
```

Модель предложила имя `publish_message` (снова в обоих случаях).
ROUGE1 = 0,4.

Неудачный пример — функция `_print_stat` (#188):
```python
def <extra_id_0>(self):
    """
    Occasionally print out stats about how fast the files are getting processed
    """
    if ((timezone.utcnow() - self.last_stat_print_time).total_seconds() >
            self.print_stats_interval):
        if len(self._file_paths) > 0:
            self._log_file_processing_stats(self._file_paths)
        self.last_stat_print_time = timezone.utcnow()
```

Без комментариев модель предложила имя `print_stats` (ROUGE1 = 0).
С комментариями — `_log_file_processing_` (ROUGE1 = 0.5).

Ещё интересный пример — функция `on_kill` (#416).
Предсказание модели оказалось лучше на данных без комментариев.
Предсказанное имя с комментариями — `cancel_query`,
а без них — `kill_query`.
Кажется, что вариант, предложенный моделью лучше, варианта из исходного кода.
```python
def <extra_id_0>(self):
    """
        Cancel the submitted athena query
        """
    if self.query_execution_id:
        self.log.info('⚰️⚰️⚰️ Received a kill Signal. Time to Die')
        self.log.info(
            'Stopping Query with executionId - %s', self.query_execution_id
        )
        response = self.hook.stop_query(self.query_execution_id)
        http_status_code = None
        try:
            http_status_code = response['ResponseMetadata']['HTTPStatusCode']
            except Exception as ex:
            self.log.error('Exception while cancelling query', ex)
                finally:
                if http_status_code is None or http_status_code != 200:
                self.log.error('Unable to request query cancel on athena. Exiting')
                else:
                self.log.info(
                    'Polling Athena for query with id %s to reach final state', self.query_execution_id
                )
                self.hook.poll_query_status(self.query_execution_id)
```

## Часть 3: язык Go

В качестве второго языка я выбрал Go.
В целом работа с ним была похожа на работу с Питоном, но была пара
существенных отличий.
tree-sitter для Питона использовал шаблон `function_definition`,
а для Go — `function_declaration`.
Также документация к функциям в датасете для Go хранилась отдельно
от кода функции, пришлось её добавлять вручную.

Результаты:

* EM (с комментариями): 0.746
* EM (без комментариев): 0.100
* ROUGE1 (с комментариями): 0.752
* ROUGE1 (без): 0.164

Примеры результатов:

Функция с настоящим именем `RegisterPullRequestHandler` (#689).
```go
// RegisterPullRequestHandler registers a plugin's github.PullRequestEvent handler.
func <extra_id_0>(name string, fn PullRequestHandler, help HelpProvider) {
    pluginHelp[name] = help
    pullRequestHandlers[name] = fn
}
```

Предсказание модели с комментариями — `RegisterPullRequestHandler`,
без комментариев — `initPullRequestHandlers`.
Этот пример объясняет, почему для Go получаются такие высокие значения
метрик.
В документации к функции как правило указано её имя.
В некоторых случаях, имя появляется при использовании логгера.

Тем не менее, даже так модель не всегда справляется.
Пример `getRemotePeerURLs` (#355):
```go
// getRemotePeerURLs returns peer urls of remote members in the cluster. The
// returned list is sorted in ascending lexicographical order.
func <extra_id_0>(cl *membership.RaftCluster, local string) []string {
    us := make([]string, 0)
    for _, m := range cl.Members() {
        if m.Name == local {
            continue
        }
        us = append(us, m.PeerURLs...)
    }
    sort.Strings(us)
    return us
}
```

Предсказанное с комментариями имя — `Copyright` (???),
без них — `peers`.

Также мне встречался пример, где в качестве имени модель генерировала
фрагмент ссылки на гитхаб.
К сожалению, я его не сохранил.
В целом, генерация имён для функций для функций Go работала хуже, чем
в случае Питона (если, конечно, имя функции не было указано в комментарии).
