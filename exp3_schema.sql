-- MySQL dump 10.13  Distrib 5.5.22, for debian-linux-gnu (i686)
--
-- Host: localhost    Database: exp3
-- ------------------------------------------------------
-- Server version	5.5.22-0ubuntu1

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
-- Table structure for table `RFeature`
--

DROP TABLE IF EXISTS `RFeature`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `RFeature` (
  `trial` smallint(4) unsigned NOT NULL,
  `stack` smallint(4) unsigned NOT NULL,
  `version` tinyint(2) unsigned NOT NULL,
  `probe` int(10) unsigned NOT NULL,
  `frame` int(6) unsigned zerofill NOT NULL DEFAULT '000000',
  `x` float NOT NULL,
  `y` float NOT NULL,
  `mass` float NOT NULL,
  `size` float NOT NULL,
  `ecc` float NOT NULL,
  `timestamp` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (`trial`,`stack`,`version`,`probe`,`frame`)
) ENGINE=MyISAM DEFAULT CHARSET=latin1;
/*!40101 SET character_set_client = @saved_cs_client */;

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
  `video` smallint(4) unsigned DEFAULT NULL,
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
) ENGINE=InnoDB AUTO_INCREMENT=41 DEFAULT CHARSET=latin1;
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
/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;

-- Dump completed on 2012-05-08 14:00:43
