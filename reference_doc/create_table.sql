CREATE TABLE `modelcache_llm_answer` (
  `id` bigint(20) unsigned NOT NULL AUTO_INCREMENT comment '主键',
  `gmt_create` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP comment '创建时间',
  `gmt_modified` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP comment '修改时间',
  `question` text NOT NULL comment 'question',
  `answer` text NOT NULL comment 'answer',
  `answer_type` int(11) NOT NULL comment 'answer_type',
  `hit_count` int(11) NOT NULL DEFAULT '0' comment 'hit_count',
  `model` varchar(1000) NOT NULL comment 'model',
  `embedding_data` blob NOT NULL comment 'embedding_data',
  `is_deleted` tinyint(1) NOT NULL DEFAULT '0' COMMENT 'delete state(0 Not deleted,-1 deleted)',
  PRIMARY KEY(`id`)
) AUTO_INCREMENT = 1 DEFAULT CHARSET = utf8mb4 COMMENT = 'cache_codegpt_answer';


CREATE TABLE `modelcache_query_log` (
  `id` bigint(20) unsigned NOT NULL AUTO_INCREMENT comment '主键',
  `gmt_create` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP comment '创建时间',
  `gmt_modified` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP comment '修改时间',
  `error_code` int(11) NOT NULL comment 'errorCode',
  `error_desc` varchar(1000) NOT NULL comment 'errorDesc',
  `cache_hit` varchar(100) NOT NULL comment 'cacheHit',
  `delta_time` float NOT NULL comment 'delta_time',
  `model` varchar(1000) NOT NULL comment 'model',
  `query` text NOT NULL comment 'query',
  `hit_query` text NOT NULL comment 'hitQuery',
  `answer` text NOT NULL comment 'answer',
  PRIMARY KEY(`id`)
) AUTO_INCREMENT = 1 DEFAULT CHARSET = utf8mb4 COMMENT = 'modelcache_query_log';
