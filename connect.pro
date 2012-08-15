function openconn 
	get_sql_setup, user, pass, driver, url, protocol, host, port, database_version
        obj = OBJ_NEW('IDLJavaObject$idl_sql','idl_sql')
        CATCH, error_status
        IF (error_status NE 0) THEN BEGIN
                catch, /cancel
                obj_destroy, obj
                message, get_sql_catch()
        ENDIF
	; print url
	obj->openPersistentConn,url,user,pass,driver
	return, obj
end

pro closeconn, obj
	obj_destroy, obj
end
