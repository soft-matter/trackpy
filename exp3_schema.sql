-- MySQL dump 10.13  Distrib 5.5.24, for debian-linux-gnu (i686)
--
-- Host: localhost    Database: exp3
-- ------------------------------------------------------
-- Server version	5.5.24-0ubuntu0.12.04.1

/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8 */;
/*!40103 SET @OLD_TIME_ZONE=@@TIME_ZONE */;
/*!40103 SET TIME_ZONE='+00:00' */;
/*!40014 SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0 */;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;
/*!40111 SET @OLD_SQL_NOTES=@@SQL_NOTES, SQL_NOTES=0 */;

--
-- Table structure for table `Batch`
--

DROP TABLE IF EXISTS `Batch`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `Batch` (
  `batch` int(10) unsigned NOT NULL AUTO_INCREMENT,
  `trial` smallint(5) unsigned DEFAULT NULL,
  `stack` smallint(5) unsigned DEFAULT NULL,
  `timestamp` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  `comment` varchar(255) DEFAULT NULL,
  PRIMARY KEY (`batch`)
) ENGINE=InnoDB AUTO_INCREMENT=10 DEFAULT CHARSET=latin1;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `MSD`
--

DROP TABLE IF EXISTS `MSD`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `MSD` (
  `stack` int(10) unsigned NOT NULL DEFAULT '0',
  `trial` int(10) unsigned NOT NULL DEFAULT '0',
  `probe` int(10) unsigned NOT NULL,
  `datapoint_num` mediumint(8) unsigned NOT NULL,
  `t` float NOT NULL,
  `x` float NOT NULL,
  `y` float NOT NULL,
  `x2` float NOT NULL,
  `y2` float NOT NULL,
  `r2` float NOT NULL,
  `N` int(10) unsigned NOT NULL,
  `timestamp` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (`trial`,`stack`,`probe`,`datapoint_num`)
) ENGINE=MyISAM AUTO_INCREMENT=4120 DEFAULT CHARSET=latin1;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `Probe`
--

DROP TABLE IF EXISTS `Probe`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `Probe` (
  `trial` smallint(5) unsigned NOT NULL,
  `stack` smallint(5) unsigned NOT NULL,
  `version` tinyint(3) unsigned NOT NULL,
  `probe` int(10) unsigned NOT NULL,
  `fit_A` float DEFAULT NULL,
  `fit_n` float DEFAULT NULL,
  `timestamp` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (`trial`,`stack`,`version`,`probe`)
) ENGINE=MyISAM DEFAULT CHARSET=latin1;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Temporary table structure for view `ProbeAnalytics`
--

DROP TABLE IF EXISTS `ProbeAnalytics`;
/*!50001 DROP VIEW IF EXISTS `ProbeAnalytics`*/;
SET @saved_cs_client     = @@character_set_client;
SET character_set_client = utf8;
/*!50001 CREATE TABLE `ProbeAnalytics` (
  `trial` smallint(4) unsigned,
  `stack` smallint(2) unsigned,
  `version` tinyint(3) unsigned,
  `age` time,
  `total` bigint(21),
  `localized` bigint(21),
  `subdiffusive` bigint(21),
  `diffusive` bigint(21),
  `G_0 [pN/μm]` double(20,3),
  `δG_0` double(20,3),
  `D [μm^2/s]` double(20,3),
  `δD` double(20,3)
) ENGINE=MyISAM */;
SET character_set_client = @saved_cs_client;

--
-- Table structure for table `Stack`
--

DROP TABLE IF EXISTS `Stack`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `Stack` (
  `trial` smallint(4) unsigned NOT NULL,
  `stack` smallint(2) unsigned NOT NULL AUTO_INCREMENT,
  `start` time DEFAULT NULL,
  `end` time DEFAULT NULL,
  `vstart` time DEFAULT NULL,
  `vduration` time DEFAULT NULL,
  `timestamp` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `status` enum('reserved','muxing started','muxed') DEFAULT NULL,
  `video` varchar(31) DEFAULT NULL,
  `comment` varchar(1023) DEFAULT NULL,
  PRIMARY KEY (`trial`,`stack`)
) ENGINE=MyISAM DEFAULT CHARSET=latin1;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `Trial`
--

DROP TABLE IF EXISTS `Trial`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `Trial` (
  `trial` smallint(4) unsigned NOT NULL AUTO_INCREMENT,
  `description` varchar(63) DEFAULT NULL,
  `who` varchar(63) DEFAULT 'Dan Allan',
  `timestamp` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `probes` varchar(1023) DEFAULT NULL,
  `comment` varchar(1023) DEFAULT NULL,
  PRIMARY KEY (`trial`)
) ENGINE=InnoDB AUTO_INCREMENT=56 DEFAULT CHARSET=latin1;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `UFeature`
--

DROP TABLE IF EXISTS `UFeature`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `UFeature` (
  `trial` smallint(4) unsigned NOT NULL DEFAULT '0',
  `stack` smallint(2) unsigned NOT NULL DEFAULT '0',
  `version` tinyint(2) unsigned NOT NULL DEFAULT '0',
  `frame` int(6) unsigned zerofill NOT NULL,
  `feature` smallint(3) unsigned NOT NULL AUTO_INCREMENT,
  `x` float NOT NULL,
  `y` float NOT NULL,
  `mass` float NOT NULL,
  `size` float NOT NULL,
  `ecc` float NOT NULL,
  `timestamp` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (`trial`,`stack`,`version`,`frame`,`feature`)
) ENGINE=MyISAM DEFAULT CHARSET=latin1;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Final view structure for view `ProbeAnalytics`
--

/*!50001 DROP TABLE IF EXISTS `ProbeAnalytics`*/;
/*!50001 DROP VIEW IF EXISTS `ProbeAnalytics`*/;
/*!50001 SET @saved_cs_client          = @@character_set_client */;
/*!50001 SET @saved_cs_results         = @@character_set_results */;
/*!50001 SET @saved_col_connection     = @@collation_connection */;
/*!50001 SET character_set_client      = utf8 */;
/*!50001 SET character_set_results     = utf8 */;
/*!50001 SET collation_connection      = utf8_general_ci */;
/*!50001 CREATE ALGORITHM=UNDEFINED */
/*!50013 DEFINER=`root`@`localhost` SQL SECURITY DEFINER */
/*!50001 VIEW `ProbeAnalytics` AS select `Stack`.`trial` AS `trial`,`Stack`.`stack` AS `stack`,`Probe`.`version` AS `version`,`Stack`.`start` AS `age`,count(0) AS `total`,count((case when (`Probe`.`fit_n` < 0.25) then 1 end)) AS `localized`,count((case when (`Probe`.`fit_n` between 0.25 and 0.85) then 1 end)) AS `subdiffusive`,count((case when (`Probe`.`fit_n` > 0.85) then 1 end)) AS `diffusive`,round((4.2e-3 / avg((case when (`Probe`.`fit_n` < 0.25) then `Probe`.`fit_A` end))),3) AS `G_0 [pN/μm]`,round((4.2e-3 / exp(std((case when (`Probe`.`fit_n` < 0.25) then log(`Probe`.`fit_A`) end)))),3) AS `δG_0`,round((avg((case when (`Probe`.`fit_n` > 0.85) then `Probe`.`fit_A` end)) / 4),3) AS `D [μm^2/s]`,round((exp(std((case when (`Probe`.`fit_n` > 0.85) then log(`Probe`.`fit_A`) end))) / 4),3) AS `δD` from (`Stack` join `Probe` on(((`Stack`.`trial` = `Probe`.`trial`) and (`Stack`.`stack` = `Probe`.`stack`)))) where ((`Probe`.`fit_A` > 0.2) or (`Probe`.`fit_n` > 0.1)) group by `Stack`.`trial`,`Stack`.`stack`,`Probe`.`version` order by `Stack`.`trial`,`Stack`.`start` */;
/*!50001 SET character_set_client      = @saved_cs_client */;
/*!50001 SET character_set_results     = @saved_cs_results */;
/*!50001 SET collation_connection      = @saved_col_connection */;
/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;

-- Dump completed on 2012-08-29 13:36:10
